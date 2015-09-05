//Copyright (C) 2011  Carl Rogers
//Released under MIT License
//license available in LICENSE file, or at http://www.opensource.org/licenses/mit-license.php

#include "cnpy.h"

#include <complex>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <stdexcept>
#include <sstream>
#include <cstdio>
#include <cassert>
#include <fstream>

#include <zlib.h>
#include <zip.h>
#include <zzip/zzip.h>

typedef std::pair<std::string, cnpy::NpArray> NpArrayDictItem;

static int closefile(int fp)
{
    return ::close(fp);
}

static int closefile(std::FILE* fp)
{
    return std::fclose(fp);
}

static int closefile(struct zip* fp)
{
    return zip_close(fp);
}

static int closefile(struct zip_file* fp)
{
    return zip_fclose(fp);
}

static int closefile(struct zip_source* fp)
{
    // zip_source is reference counted.
    //zip_source_free(fp);
    return 0;
}

template<typename _Tp>
class Handler
{
public:
    Handler(_Tp* z=nullptr): h(z) {}
    ~Handler() { if(h) closefile(h); }
    _Tp* handle() { return h; }
    Handler& operator=(const _Tp& z)
    {
        this->h = z;
        return *this;
    }
    void close() { closefile(h); h = nullptr; }
private:
    _Tp* h=nullptr;
};


struct NpHeader_
{
    uint8_t x93 = 0x93;
    int8_t numpy[5] = {'N', 'U', 'M', 'P', 'Y'};
    uint8_t majorVersion = 0x01;
    uint8_t minorVersion = 0x00;
    uint16_t dictSize = 0;
};

typedef struct NpHeader_ NpHeader;

static_assert(sizeof(NpHeader)==(1+5+1+1+2), "The npy header struct must be 10 bytes");

static char BigEndianTest()
{
    unsigned char x[] = {1, 0};
    short y = *(short*)x;
    return y == 1 ? '<' : '>';
}


static char map_type(const cnpy::Type& t)
{
    switch (t)
    {
    case cnpy::Type::Int8:
    case cnpy::Type::Int16:
    case cnpy::Type::Int32:
    case cnpy::Type::Int64:
        return 'i';
    case cnpy::Type::Uint8:
    case cnpy::Type::Uint16:
    case cnpy::Type::Uint32:
    case cnpy::Type::Uint64:
        return 'u';
    case cnpy::Type::Float:
    case cnpy::Type::Double:
    case cnpy::Type::LongDouble:
        return 'f';
    case cnpy::Type::ComplexFloat:
    case cnpy::Type::ComplexDouble:
    case cnpy::Type::ComplexLongDouble:
        return 'c';
    case cnpy::Type::Bool:
        return 'b';
    default:
        return '?';
    }
}

template<typename T> static std::vector<char>& operator+=(std::vector<char>& lhs, const T rhs) {
    //write in little endian
    for(char byte = 0; byte < sizeof(T); byte++) {
        char val = *((char*)&rhs+byte);
        lhs.push_back(val);
    }
    return lhs;
}


template<> std::vector<char>& operator+=(std::vector<char>& lhs, const std::string rhs)
{
    lhs.insert(lhs.end(),rhs.begin(),rhs.end());
    return lhs;
}


template<> std::vector<char>& operator+=(std::vector<char>& lhs, const char* rhs)
{
    //write in little endian
    size_t len = strlen(rhs);
    lhs.reserve(len);
    for(size_t byte = 0; byte < len; byte++) {
        lhs.push_back(rhs[byte]);
    }
    return lhs;
}


template<typename T> std::string tostring(T i, int pad = 0, char padval = ' ')
{
    std::stringstream s;
    s << i;
    return s.str();
}

static std::vector<char> create_npy_header(const cnpy::Type dtype,
                                           const size_t elementSize,
                                           const std::vector<size_t>& shape)
{
    size_t ndims = shape.size();

    std::string dict;
    dict += "{'descr': '";
    dict += BigEndianTest();
    dict += map_type(dtype);
    dict += tostring(elementSize);
    dict += "', 'fortran_order': False, 'shape': (";
    dict += tostring(shape[0]);
    for(int i=1; i<ndims; i++)
        dict += ", " + tostring(shape[i]);
    if(ndims == 1)
        dict += ",";
    dict += "), }";
    //pad with spaces so that preamble+dict is modulo 16 bytes. preamble is 10 bytes. dict needs to end with \n
    int remainder = 16 - (10 + dict.size()) % 16;
    //dict.insert(dict.end(), remainder, ' ');
    for(int i=0; i<remainder-1; ++i)
        dict += ' ';
    dict += '\n';

    NpHeader header;
    header.dictSize = dict.size();

    std::vector<char> bytes;
    bytes.resize(sizeof(NpHeader) + dict.size());
    std::memcpy(bytes.data(), &header, sizeof(NpHeader));
    std::memcpy(bytes.data()+sizeof(NpHeader), dict.data(), dict.size());

    return bytes;
}


static void parse_npy_header(std::FILE* fp, size_t& word_size, std::vector<size_t>& shape, bool& fortran_order)
{
    char buffer[256];
    size_t res = std::fread(buffer, sizeof(char), 11, fp);
    if(res != 11)
        throw std::runtime_error("parse_npy_header: failed fread");
    const std::string header = std::fgets(buffer, 256, fp);
    assert(header[header.size()-1] == '\n');

    int loc1, loc2;

    //fortran order
    loc1 = header.find("fortran_order")+16;
    fortran_order = (header.substr(loc1,5) == "True" ? true : false);

    //shape
    loc1 = header.find("(");
    loc2 = header.find(")");
    std::string str_shape = header.substr(loc1+1,loc2-loc1-1);
    int ndims = 0;
    if(str_shape[str_shape.size()-1] == ',')
        ndims = 1;
    else
        ndims = std::count(str_shape.begin(), str_shape.end(), ',') + 1;
    shape.resize(ndims);
    for(int i = 0;i < ndims; i++)
    {
        loc1 = str_shape.find(",");
        shape[i] = std::stoull(str_shape.substr(0, loc1));
        str_shape = str_shape.substr(loc1+1);
    }

    //endian, word size, data type
    //byte order code | stands for not applicable. 
    //not sure when this applies except for byte array
    loc1 = header.find("descr")+9;
    bool littleEndian = header[loc1] == '<' || header[loc1] == '|';//;) // ? true : false);
    assert(littleEndian);

    //char type = header[loc1+1];

    std::string str_ws = header.substr(loc1+2);
    loc2 = str_ws.find("'");
    word_size = std::stoull(str_ws.substr(0,loc2));
}

static void parseDictHeader(const std::string& dict, size_t& word_size, std::vector<size_t>& shape, bool& fortran_order)
{
    int loc1, loc2;

    //fortran order
    loc1 = dict.find("fortran_order") + 16;
    fortran_order = dict.substr(loc1, 5) == "True";

    //shape
    loc1 = dict.find("(");
    loc2 = dict.find(")");
    std::string str_shape = dict.substr(loc1+1, loc2-loc1-1);
    int ndims = 0;
    if(str_shape[str_shape.size()-1] == ',')
        ndims = 1;
    else
        ndims = std::count(str_shape.begin(), str_shape.end(), ',') + 1;
    shape.resize(ndims);
    for(int i = 0;i < ndims; i++)
    {
        loc1 = str_shape.find(",");
        shape[i] = std::stoull(str_shape.substr(0, loc1));
        str_shape = str_shape.substr(loc1+1);
    }

    //endian, word size, data type
    //byte order code | stands for not applicable.
    //not sure when this applies except for byte array
    loc1 = dict.find("descr")+9;
    bool littleEndian = dict[loc1] == '<' || dict[loc1] == '|';
    if(!littleEndian)
        throw std::runtime_error("Big endian data can note be managed.");

    //char type = dict[loc1+1];

    std::string str_ws = dict.substr(loc1+2);
    loc2 = str_ws.find("'");
    word_size = std::stoull(str_ws.substr(0, loc2));
}


static void parse_zip_footer(FILE* fp, unsigned short& nrecs, unsigned int& global_header_size, unsigned int& global_header_offset)
{
    std::vector<char> footer(22);
    fseek(fp,-22,SEEK_END);
    size_t res = fread(&footer[0],sizeof(char),22,fp);
    if(res != 22)
        throw std::runtime_error("parse_zip_footer: failed fread");

    unsigned short disk_no, disk_start, nrecs_on_disk, comment_len;
    disk_no = *(unsigned short*) &footer[4];
    disk_start = *(unsigned short*) &footer[6];
    nrecs_on_disk = *(unsigned short*) &footer[8];
    nrecs = *(unsigned short*) &footer[10];
    global_header_size = *(unsigned int*) &footer[12];
    global_header_offset = *(unsigned int*) &footer[16];
    comment_len = *(unsigned short*) &footer[20];

    assert(disk_no == 0);
    assert(disk_start == 0);
    assert(nrecs_on_disk == nrecs);
    assert(comment_len == 0);
}


static cnpy::NpArray load_the_npy_file(std::FILE* fp)
{
    std::vector<size_t> shape;
    size_t word_size;
    bool fortran_order;
    parse_npy_header(fp, word_size, shape, fortran_order);

    cnpy::NpArray arr(shape, word_size, fortran_order);

    const size_t num = arr.numElements();
    const size_t nread = std::fread(arr.data(), word_size, num, fp);
    if(nread != num)
        throw std::runtime_error("npy file read error: expected "+std::to_string(arr.size())+", read "+std::to_string(nread));

    return arr;
}

static cnpy::NpArray load_the_npy_file(Handler<struct zip_file>& zipFile)
{
    std::vector<size_t> shape;
    size_t word_size;
    bool fortran_order;

    NpHeader header;
    zip_int64_t nread = zip_fread(zipFile.handle(), &header, sizeof(NpHeader));
    if(nread != sizeof(NpHeader))
         throw std::runtime_error("Error reading npy header in npz file");

    std::string dict;
    dict.resize(header.dictSize, ' ');
    nread = zip_fread(zipFile.handle(), &dict[0], header.dictSize);
    if(nread != header.dictSize)
         throw std::runtime_error("Error reading npy dict header in npz file");

    parseDictHeader(dict, word_size, shape, fortran_order);

    cnpy::NpArray arr(shape, word_size, fortran_order);

    nread = zip_fread(zipFile.handle(), arr.data(), arr.size());
    if(nread != arr.size())
        throw std::runtime_error("npy file read error: expected "+std::to_string(arr.size())+", read "+std::to_string(nread));

    return arr;
}


void cnpy::npy_save_data(const std::string& fname,
                         const unsigned char* data, const Type dtype,
                         const size_t elemSize, const std::vector<size_t>& shape,
                         const char mode)
{
    FILE* fp = NULL;

    if(mode == 'a')
        fp = fopen(fname.c_str(),"r+b");

    if(fp)
    {
        //file exists. we need to append to it. read the header, modify the array size
        size_t word_size;
        std::vector<size_t> tmp_shape;
        bool fortran_order;
        parse_npy_header(fp, word_size, tmp_shape, fortran_order);
        assert(!fortran_order);

        if(word_size != elemSize) {
            std::cout<<"libnpy error: "<<fname<<" has word size "<<word_size<<" but npy_save appending data sized "<<elemSize<<"\n";
            assert( word_size == elemSize );
        }
        if(tmp_shape.size() != shape.size())
            throw std::runtime_error("Attempting to append misdimensioned data to "+fname);

        for(int i=1; i<shape.size(); ++i)
        {
            if(shape[i] != tmp_shape[i])
                throw std::runtime_error("Attempting to append misshaped data to "+fname);
        }
        tmp_shape[0] += shape[0];

        fseek(fp, 0, SEEK_SET);
        std::vector<char> header = create_npy_header(dtype, elemSize, tmp_shape);
        fwrite(header.data(), sizeof(char), header.size(), fp);
        fseek(fp, 0, SEEK_END);
    }
    else
    {
        fp = fopen(fname.c_str(),"wb");
        std::vector<char> header = create_npy_header(dtype, elemSize, shape);
        fwrite(header.data(), sizeof(char), header.size(), fp);
    }

    size_t nels = std::accumulate(shape.cbegin(), shape.cend(), 1U, std::multiplies<size_t>());
    std::fwrite(data, elemSize, nels, fp);
    fclose(fp);
}

/*
void cnpy::npz_save_data(const std::string& zipname, const std::string& name,
                         const unsigned char* data, const cnpy::Type dtype,
                         const size_t elemSize, const std::vector<size_t>& shape,
                         const char mode)
{
    //first, append a .npy to the fname
    std::string fname(name);
    fname += ".npy";

    //now, on with the show
    FILE* fp = NULL;
    unsigned short nrecs = 0;
    unsigned int global_header_offset = 0;
    std::vector<char> global_header;

    if(mode == 'a')
        fp = fopen(zipname.c_str(), "r+b");

    if(fp)
    {
        //zip file exists. we need to add a new npy file to it.
        //first read the footer. this gives us the offset and size of the global header
        //then read and store the global header.
        //below, we will write the the new data at the start of the global header then append the global header and footer below it
        unsigned int global_header_size;
        parse_zip_footer(fp, nrecs, global_header_size, global_header_offset);
        fseek(fp, global_header_offset, SEEK_SET);
        global_header.resize(global_header_size);
        size_t res = fread(global_header.data(), sizeof(char), global_header_size, fp);
        if(res != global_header_size)
            throw std::runtime_error("npz_save: header read error while adding to existing zip");

        fseek(fp, global_header_offset, SEEK_SET);
    }
    else
    {
        fp = fopen(zipname.c_str(), "wb");
    }

    std::vector<char> npy_header = create_npy_header(dtype, elemSize, shape);

    size_t nels = std::accumulate(shape.cbegin(), shape.cend(), 1U, std::multiplies<size_t>());
    size_t nbytes = nels*elemSize + npy_header.size();

    //get the CRC of the data to be added
    unsigned int crc = crc32(0L, (unsigned char*)npy_header.data(), npy_header.size());
    crc = crc32(crc, data, nels*elemSize);

    //build the local header
    std::vector<char> local_header;
    local_header += "PK"; //first part of sig
    local_header += (unsigned short) 0x0403; //second part of sig
    local_header += (unsigned short) 20; //min version to extract
    local_header += (unsigned short) 0; //general purpose bit flag
    local_header += (unsigned short) 0; //compression method
    local_header += (unsigned short) 0; //file last mod time
    local_header += (unsigned short) 0;     //file last mod date
    local_header += (unsigned int) crc; //crc
    local_header += (unsigned int) nbytes; //compressed size
    local_header += (unsigned int) nbytes; //uncompressed size
    local_header += (unsigned short) fname.size(); //fname length
    local_header += (unsigned short) 0; //extra field length
    local_header += fname;

    //build global header
    global_header += "PK"; //first part of sig
    global_header += (unsigned short) 0x0201; //second part of sig
    global_header += (unsigned short) 20; //version made by
    global_header.insert(global_header.end(),local_header.begin()+4,local_header.begin()+30);
    global_header += (unsigned short) 0; //file comment length
    global_header += (unsigned short) 0; //disk number where file starts
    global_header += (unsigned short) 0; //internal file attributes
    global_header += (unsigned int) 0; //external file attributes
    global_header += (unsigned int) global_header_offset; //relative offset of local file header, since it begins where the global header used to begin
    global_header += fname;

    //build footer
    std::vector<char> footer;
    footer += "PK"; //first part of sig
    footer += (unsigned short) 0x0605; //second part of sig
    footer += (unsigned short) 0; //number of this disk
    footer += (unsigned short) 0; //disk where footer starts
    footer += (unsigned short) (nrecs+1); //number of records on this disk
    footer += (unsigned short) (nrecs+1); //total number of records
    footer += (unsigned int) global_header.size(); //nbytes of global headers
    footer += (unsigned int) (global_header_offset + nbytes + local_header.size()); //offset of start of global headers, since global header now starts after newly written array
    footer += (unsigned short) 0; //zip file comment length

    //write everything
    fwrite(local_header.data(), sizeof(char), local_header.size(), fp);
    fwrite(npy_header.data(), sizeof(char), npy_header.size(), fp);
    fwrite(data, elemSize, nels, fp);
    fwrite(global_header.data(), sizeof(char), global_header.size(), fp);
    fwrite(footer.data(), sizeof(char), footer.size(), fp);
    fclose(fp);
}
*/

class ZipSourceCallbackData
{
public:
    ZipSourceCallbackData(const std::vector<char>& h, const unsigned char* d, const size_t dSize):
        header(h),
        data(d),
        dataSize(dSize),
        offset(0),
        headerIsDone(false)
    {}

    const std::vector<char> header;
    const unsigned char* data;
    const size_t dataSize;

    size_t offset = 0;
    bool headerIsDone = false;
};

static zip_int64_t zipSourceCallback(void* userdata, void* data, zip_uint64_t len, zip_source_cmd cmd)
{
    ZipSourceCallbackData* cbData = reinterpret_cast<ZipSourceCallbackData*>(userdata);
    switch (cmd) {
    case ZIP_SOURCE_OPEN:
        std::cout<<"ZIP_SOURCE_OPEN"<<std::endl;
        cbData->offset = 0;
        cbData->headerIsDone = false;
        std::cout<<"\t"<<cbData->header.size()<<" "<<cbData->dataSize<<std::endl;
        break;
    case ZIP_SOURCE_READ: {
        std::cout<<"ZIP_SOURCE_READ "<<len<<" ";
        if(cbData->headerIsDone)
        {
            size_t remain = cbData->dataSize-cbData->offset;
            size_t toCopy = std::min((size_t)len, remain);
            if(toCopy>0)
                std::memcpy(data, &cbData->data[cbData->offset], toCopy);
            cbData->offset += toCopy;
            std::cout<<toCopy<<" "<<cbData->offset<<" d"<<std::endl;
            return toCopy;
        }
        else
        {
            size_t remain = cbData->header.size()-cbData->offset;
            size_t toCopy = std::min((size_t)len, remain);
            if(toCopy>0)
                std::memcpy(data, &cbData->header[cbData->offset], toCopy);
            cbData->offset += toCopy;
            if(cbData->offset==cbData->header.size())
            {
                cbData->headerIsDone = true;
                cbData->offset = 0;
            }
            std::cout<<toCopy<<" "<<cbData->offset<<" h"<<std::endl;
            return toCopy;
        }
    }
    case ZIP_SOURCE_CLOSE:
        std::cout<<"ZIP_SOURCE_CLOSE"<<std::endl;
        break;
    case ZIP_SOURCE_STAT: {
        std::cout<<"ZIP_SOURCE_STAT"<<std::endl;
        struct zip_stat* zs = reinterpret_cast<struct zip_stat*>(data);
        zip_stat_init(zs);
        zs->encryption_method = ZIP_EM_NONE;
        //zs->index;			/* index within archive */
        zs->size = cbData->dataSize + cbData->header.size(); /* size of file (uncompressed) */
        //zs->comp_size;		/* size of file (compressed) */
        zs->mtime = std::time(nullptr);
        //zs->crc;			/* crc of file data */
        zs->comp_method = ZIP_CM_STORE; /* compression method used */
        zs->valid = ZIP_STAT_SIZE | ZIP_STAT_MTIME | ZIP_STAT_COMP_METHOD | ZIP_STAT_ENCRYPTION_METHOD;
        return 0;
    }
    case ZIP_SOURCE_ERROR:
        std::cout<<"ZIP_SOURCE_ERROR"<<std::endl;
        break;
    default:
        break;
    }

    return 0;
}


void cnpy::npz_save_data(const std::string& zipname, const std::string& name,
                         const unsigned char* data, const cnpy::Type dtype,
                         const size_t elemSize, const std::vector<size_t>& shape,
                         const char mode)
{
    //first, append a .npy to the fname
    std::string fname(name);
    fname += ".npy";

    if(mode=='w' && std::ifstream(zipname).is_open())
    {
        // Remove the old file if present
        if(std::remove(zipname.c_str())!=0)
            throw std::runtime_error("Unable to overwrite "+zipname);
    }

    Handler<struct zip> zip = zip_open(zipname.c_str(), ZIP_CREATE, nullptr);
    if(zip.handle()==nullptr)
        throw std::runtime_error("Error opening npz file "+zipname);

    // Remove the old array if present
    int nameLookup = zip_name_locate(zip.handle(), fname.c_str(), 0);
    if(nameLookup>=0 && zip_delete(zip.handle(), nameLookup)!=0)
        throw std::runtime_error("Unable to overwrite "+name+" array");

    std::vector<char> header = create_npy_header(dtype, elemSize, shape);

    const int dataSize = std::accumulate(shape.cbegin(), shape.cend(), elemSize, std::multiplies<size_t>());
    ZipSourceCallbackData cbData(header, data, dataSize);

    Handler<struct zip_source> zipSource = zip_source_function(zip.handle(), zipSourceCallback, &cbData);
    if(zipSource.handle()==nullptr)
        throw std::runtime_error("Error creating "+name+" array");

    zip_int64_t fid = zip_add(zip.handle(), fname.c_str(), zipSource.handle());
    if(fid<0)
    {
        zip_source_free(zipSource.handle());
        throw std::runtime_error("Error creating "+name+" array");
    }

    zip.close();
}



cnpy::NpArrayDict cnpy::npz_load(const std::string& fname)
{
    Handler<struct zip> zip = zip_open(fname.c_str(), ZIP_CHECKCONS, nullptr);
    if(zip.handle()==nullptr)
        throw std::runtime_error("Error opening npz file "+fname);

    NpArrayDict arrays;
    zip_uint64_t numFiles = zip_get_num_entries(zip.handle(), 0);
    for(zip_uint64_t fid=0; fid<numFiles; ++fid)
    {
        const char* arrName = zip_get_name(zip.handle(), fid, 0);
        if(arrName==nullptr)
            continue;

        Handler<struct zip_file> zipFile = zip_fopen_index(zip.handle(), fid, 0);

        std::string name = arrName;
        name.erase(name.size()-4);
        arrays.insert(NpArrayDictItem(name, load_the_npy_file(zipFile)));
    }

    return arrays;
}


cnpy::NpArray cnpy::npz_load(const std::string& fname, const std::string& varname)
{
    Handler<struct zip> zip = zip_open(fname.c_str(), ZIP_CHECKCONS, nullptr);
    if(zip.handle()==nullptr)
        throw std::runtime_error("Error opening npz file "+fname);

    std::string key = varname + ".npy";
    int nameLookup = zip_name_locate(zip.handle(), key.c_str(), 0);
    if(nameLookup<0)
        throw std::runtime_error("Variable name "+varname+" not found in "+fname);

    Handler<struct zip_file> zipFile = zip_fopen_index(zip.handle(), nameLookup, 0);

    NpArray array = load_the_npy_file(zipFile);
    return array;
}

cnpy::NpArray cnpy::npy_load(const std::string& fname)
{
    FILE* fp = fopen(fname.c_str(), "rb");

    if(!fp) {
        printf("npy_load: Error! Unable to open file %s!\n",fname.c_str());
        return NpArray();
    }

    NpArray arr = load_the_npy_file(fp);

    fclose(fp);
    return arr;
}




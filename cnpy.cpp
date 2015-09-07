//Copyright (C) 2011  Carl Rogers
//Released under MIT License
//license available in LICENSE file, or at http://www.opensource.org/licenses/mit-license.php

/*
 * The orignal cnpy library was modified by
 * Alessandro Bacchini, allebacco@gmail.com
 */

#include "cnpy.h"

#include <cstdlib>
#include <iomanip>
#include <stdexcept>
#include <sstream>
#include <cstdio>
#include <cassert>
#include <fstream>
#include <iostream>

#include <zip.h>


typedef std::pair<std::string, cnpy::NpArray> NpArrayDictItem;


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

static cnpy::Type descr2Type(const char c, const size_t byteSize)
{
    std::cout<<"descr2Type "<<c<<" "<<byteSize<<std::endl;
    switch (c)
    {
    case 'i':
        switch (byteSize)
        {
        case sizeof(int8_t): return cnpy::Type::Int8;
        case sizeof(int16_t): return cnpy::Type::Int16;
        case sizeof(int32_t): return cnpy::Type::Int32;
        case sizeof(int64_t): return cnpy::Type::Int64;
        default: return cnpy::Type::Void;
        }
        break;
    case 'u':
        switch (byteSize)
        {
        case sizeof(uint8_t): return cnpy::Type::Uint8;
        case sizeof(uint16_t): return cnpy::Type::Uint16;
        case sizeof(uint32_t): return cnpy::Type::Uint32;
        case sizeof(uint64_t): return cnpy::Type::Uint64;
        default: return cnpy::Type::Void;
        }
        break;
    case 'f':
        switch (byteSize)
        {
        case sizeof(float): return cnpy::Type::Float;
        case sizeof(double): return cnpy::Type::Double;
        case sizeof(long double): return cnpy::Type::LongDouble;
        default: return cnpy::Type::Void;
        }
        break;
    case 'c':
        switch (byteSize)
        {
        case sizeof(std::complex<float>): return cnpy::Type::ComplexFloat;
        case sizeof(std::complex<double>): return cnpy::Type::ComplexDouble;
        case sizeof(std::complex<long double>): return cnpy::Type::ComplexLongDouble;
        default: return cnpy::Type::Void;
        }
        break;
    case 'b':
        if(byteSize == sizeof(bool)) return cnpy::Type::Bool;
        else return cnpy::Type::Void;
    default:
        return cnpy::Type::Void;
    }
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

static void parseDictHeader(const std::string& dict, size_t& word_size, std::vector<size_t>& shape, bool& fortran_order, char& elType)
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

    // byte order code | stands for not applicable.
    loc1 = dict.find("descr")+9;
    bool littleEndian = dict[loc1] == '<' || dict[loc1] == '|';
    if(!littleEndian)
        throw std::runtime_error("Big endian data can note be managed.");

    elType = dict[loc1+1];

    std::string str_ws = dict.substr(loc1+2);
    loc2 = str_ws.find("'");
    word_size = std::stoull(str_ws.substr(0, loc2));
}


static cnpy::NpArray load_the_npy_file(Handler<std::FILE>& npyFile)
{
    std::vector<size_t> shape;
    size_t word_size;
    bool fortran_order;

    NpHeader header;
    zip_int64_t nread = std::fread(&header, sizeof(NpHeader), 1, npyFile.handle());
    if(nread != 1)
         throw std::runtime_error("Error reading npy header");

    std::string dict;
    dict.resize(header.dictSize, ' ');
    nread = std::fread(&dict[0], sizeof(char), header.dictSize, npyFile.handle());
    if(nread != header.dictSize)
         throw std::runtime_error("Error reading npy dict header");

    char elType;
    parseDictHeader(dict, word_size, shape, fortran_order, elType);

    cnpy::NpArray arr(shape, word_size, descr2Type(elType, word_size), fortran_order);

    nread = std::fread(arr.data(), arr.elemSize(), arr.numElements(), npyFile.handle());
    if(nread != arr.numElements())
        throw std::runtime_error("npy file read error: expected "+std::to_string(arr.numElements())+", read "+std::to_string(nread));

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

    char elType;
    parseDictHeader(dict, word_size, shape, fortran_order, elType);

    cnpy::NpArray arr(shape, word_size, descr2Type(elType, word_size), fortran_order);

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

    const std::vector<char>& header;
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
        cbData->offset = 0;
        cbData->headerIsDone = false;
        break;
    case ZIP_SOURCE_READ: {
        if(cbData->headerIsDone)
        {
            size_t remain = cbData->dataSize-cbData->offset;
            size_t toCopy = std::min((size_t)len, remain);
            if(toCopy>0)
                std::memcpy(data, &cbData->data[cbData->offset], toCopy);
            cbData->offset += toCopy;
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
            return toCopy;
        }
    }
    case ZIP_SOURCE_CLOSE:
        break;
    case ZIP_SOURCE_STAT: {
        struct zip_stat* zs = reinterpret_cast<struct zip_stat*>(data);
        zip_stat_init(zs);
        zs->encryption_method = ZIP_EM_NONE;
        zs->size = cbData->dataSize + cbData->header.size(); /* size of file (uncompressed) */
        zs->mtime = std::time(nullptr);
        zs->comp_method = ZIP_CM_STORE; /* compression method used */
        zs->valid = ZIP_STAT_SIZE | ZIP_STAT_MTIME | ZIP_STAT_COMP_METHOD | ZIP_STAT_ENCRYPTION_METHOD;
        return 0;
    }
    case ZIP_SOURCE_ERROR:
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
    Handler<std::FILE> fp = std::fopen(fname.c_str(), "r");

    NpArray arr = load_the_npy_file(fp);

    return arr;
}




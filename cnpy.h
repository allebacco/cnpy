//Copyright (C) 2011  Carl Rogers
//Released under MIT License
//license available in LICENSE file, or at http://www.opensource.org/licenses/mit-license.php

#ifndef LIBCNPY_H_
#define LIBCNPY_H_

#include <string>
#include <stdexcept>
#include <sstream>
#include <vector>
#include <cstdio>
#include <typeinfo>
#include <iostream>
#include <cassert>
#include <zlib.h>
#include <map>

namespace cnpy {

struct NpyArray {
    char* data;
    std::vector<size_t> shape;
    unsigned int word_size;
    bool fortran_order;
    void destruct() {delete[] data;}
};

struct npz_t : public std::map<std::string, NpyArray>
{
    void destruct()
    {
        npz_t::iterator it = this->begin();
        for(; it != this->end(); ++it) (*it).second.destruct();
    }
};

/*
std::vector<char> create_npy_header(const std::type_info& dataType, const size_t elementSize, const size_t* shape, const size_t ndims);
void parse_npy_header(FILE* fp, size_t& word_size, size_t*& shape, size_t& ndims, bool& fortran_order);
void parse_zip_footer(FILE* fp, unsigned short& nrecs, unsigned int& global_header_size, unsigned int& global_header_offset);
*/

npz_t npz_load(const std::string& fname);
NpyArray npz_load(const std::string& fname, const std::string& varname);
NpyArray npy_load(const std::string& fname);

void npy_save_data(const std::string& fname,
                   const unsigned char* data,const std::type_info& typeInfo,
                   const size_t elemSize, const size_t* shape, const size_t ndims,
                   const char mode='w');
void npz_save_data(const std::string& zipname, const std::string& name,
                   const unsigned char* data,const std::type_info& typeInfo,
                   const size_t elemSize, const size_t* shape, const size_t ndims,
                   const char mode='w');

/*
template<typename T> std::vector<char>& operator+=(std::vector<char>& lhs, const T rhs) {
    //write in little endian
    for(char byte = 0; byte < sizeof(T); byte++) {
        char val = *((char*)&rhs+byte);
        lhs.push_back(val);
    }
    return lhs;
}

template<> std::vector<char>& operator+=(std::vector<char>& lhs, const std::string rhs);
template<> std::vector<char>& operator+=(std::vector<char>& lhs, const char* rhs);
*/


template<typename T> void npy_save(std::string fname, const T* data, const size_t* shape, const size_t ndims, const char mode='w')
{
    npy_save_data(fname, reinterpret_cast<const unsigned char*>(data),
                  typeid(T), sizeof(T), shape, ndims, mode);
}

template<typename T> void npz_save(const std::string& zipname, const std::string& name, const T* data, const size_t* shape, const size_t ndims, const char mode='w')
{
    npz_save_data(zipname, name, reinterpret_cast<const unsigned char*>(data),
                  typeid(T), sizeof(T), shape, ndims, mode);
}

}

#endif

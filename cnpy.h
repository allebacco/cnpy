//Copyright (C) 2011  Carl Rogers
//Released under MIT License
//license available in LICENSE file, or at http://www.opensource.org/licenses/mit-license.php

#ifndef LIBCNPY_H_
#define LIBCNPY_H_

#include <cstring>
#include <string>
#include <vector>
#include <type_traits>
#include <iostream>
#include <algorithm>
#include <complex>
#include <map>

namespace cnpy
{

enum class Type
{
    Void,   //!< Void datatype with undefined size.
    Int8,   //!< Signed int (1 byte)
    Int16,  //!< Signed int (2 bytes)
    Int32,  //!< Signed int (4 bytes)
    Int64,  //!< Signed int (8 bytes)
    Uint8,  //!< Unsigned int (1 byte)
    Uint16, //!< Unsigned int (2 bytes)
    Uint32, //!< Unsigned int (4 bytes)
    Uint64, //!< Unsigned int (8 bytes)
    Float,  //!< Floating point signle precision (4 bytes)
    Double, //!< Floating point double precision (8 bytes)
    LongDouble,         //!< Floating point long double precision (>=8 bytes)
    ComplexFloat,       //!< Complex floating point signle precision (2 * 4 bytes)
    ComplexDouble,      //!< Complex floating point double precision (2 * 8 bytes)
    ComplexLongDouble,  //!< Complex floating point long double precision (2 * >=8 bytes)
    Bool    //!< Boolean (1 byte)
};

template<typename _Tp> static Type type()
{
    if(std::is_same<_Tp, int8_t>::value)
        return Type::Int8;
    if(std::is_same<_Tp, int16_t>::value)
        return Type::Int16;
    if(std::is_same<_Tp, int32_t>::value)
        return Type::Int32;
    if(std::is_same<_Tp, int64_t>::value)
        return Type::Int64;
    if(std::is_same<_Tp, uint8_t>::value)
        return Type::Uint8;
    if(std::is_same<_Tp, uint16_t>::value)
        return Type::Uint16;
    if(std::is_same<_Tp, uint32_t>::value)
        return Type::Uint32;
    if(std::is_same<_Tp, uint64_t>::value)
        return Type::Uint64;
    if(std::is_same<_Tp, float>::value)
        return Type::Float;
    if(std::is_same<_Tp, double>::value)
        return Type::Double;
    if(std::is_same<_Tp, long double>::value)
        return Type::LongDouble;
    if(std::is_same<_Tp, std::complex<float>>::value)
        return Type::ComplexFloat;
    if(std::is_same<_Tp, std::complex<double>>::value)
        return Type::ComplexDouble;
    if(std::is_same<_Tp, std::complex<long double>>::value)
        return Type::ComplexLongDouble;
    if(std::is_same<_Tp, bool>::value)
        return Type::Bool;
    return cnpy::Type::Void;
}




class NpArray
{
public:
    NpArray() :
        mData(nullptr),
        mElemSize(0),
        mDataSize(0),
        mIsFortranOrder(false),
        mHasDataOwnership(true)
    {}

    NpArray(const std::vector<size_t>& shape,
            const size_t elSize,
            const bool isFortran=false,
            const unsigned char* data=nullptr) :
        mShape(shape),
        mElemSize(elSize),
        mIsFortranOrder(isFortran),
        mHasDataOwnership(true)
    {
        mDataSize = std::accumulate(mShape.begin(), mShape.end(), mElemSize, std::multiplies<size_t>());

        mData = new unsigned char[mDataSize];

        if(data!=nullptr)
            std::memcpy(mData, data, mDataSize);
    }

    ~NpArray()
    {
        if(mHasDataOwnership && mData!=nullptr)
            delete[] mData;
    }

    NpArray(NpArray&& other)
    {
        swapData(other);
    }

    NpArray& operator=(NpArray&& other)
    {
        swapData(other);
        return *this;
    }

    unsigned char* data() { return mData; }
    const unsigned char* data() const { return mData; }

    size_t shape(const size_t i) const { return mShape[i]; }
    size_t nDims() const { return mShape.size(); }
    size_t numElements() const
    {
        return std::accumulate(mShape.begin(), mShape.end(), 1U, std::multiplies<size_t>());
    }
    size_t elemSize() const { return mElemSize; }
    size_t size() const { return mDataSize; }
    bool isFortranOrder() const { return mIsFortranOrder; }
    bool hasDataOwnership() const { return mHasDataOwnership; }

    void removeDataOwnership() { mHasDataOwnership = false; }

    bool empty() { return mData==nullptr; }

private:

    void swapData(NpArray& other)
    {
        mData = other.mData;
        other.mData = nullptr;

        std::swap(mShape, other.mShape);

        mElemSize = other.mElemSize;
        other.mElemSize = 0;

        mIsFortranOrder = other.mIsFortranOrder;
        other.mIsFortranOrder = false;

        mHasDataOwnership = other.mHasDataOwnership;
        other.mHasDataOwnership = false;
    }

    unsigned char* mData;
    std::vector<size_t> mShape;
    size_t mElemSize;
    size_t mDataSize;
    bool mIsFortranOrder;

    bool mHasDataOwnership;
};


typedef std::map<std::string, NpArray> NpArrayDict;

NpArrayDict npz_load(const std::string& fname);
NpArray npz_load(const std::string& fname, const std::string& varname);
NpArray npy_load(const std::string& fname);

void npy_save_data(const std::string& fname,
                   const unsigned char* data, const Type dtype,
                   const size_t elemSize, const std::vector<size_t>& shape,
                   const char mode='w');
void npz_save_data(const std::string& zipname, const std::string& name,
                   const unsigned char* data, const Type dtype,
                   const size_t elemSize, const std::vector<size_t>& shape,
                   const char mode='w');


template<typename _Tp> void npy_save(std::string fname,
                                     const _Tp* data, const std::vector<size_t>& shape,
                                     const char mode='w')
{
    npy_save_data(fname, reinterpret_cast<const unsigned char*>(data),
                  type<_Tp>(), sizeof(_Tp), shape, mode);
}

template<typename _Tp> void npz_save(const std::string& zipname, const std::string& name,
                                   const _Tp* data, const std::vector<size_t>& shape,
                                   const char mode='w')
{
    npz_save_data(zipname, name, reinterpret_cast<const unsigned char*>(data),
                  type<_Tp>(), sizeof(_Tp), shape, mode);
}

}

#endif

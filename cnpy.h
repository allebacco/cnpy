//Copyright (C) 2011  Carl Rogers
//Released under MIT License
//license available in LICENSE file, or at http://www.opensource.org/licenses/mit-license.php

#ifndef LIBCNPY_H_
#define LIBCNPY_H_

#include <cstring>
#include <string>
#include <vector>
#include <typeinfo>
#include <iostream>
#include <algorithm>

#include <map>

namespace cnpy
{

class NpArray
{
public:
    NpArray() :
        mData(nullptr),
        mElemSize(0),
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
        size_t dataSize = std::accumulate(mShape.begin(), mShape.end(), mElemSize, std::multiplies<size_t>());

        mData = new unsigned char[dataSize];

        if(data!=nullptr)
            std::memcpy(mData, data, dataSize);
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
        return std::accumulate(mShape.begin(), mShape.end(), 1, std::multiplies<size_t>());
    }
    size_t elemSize() const { return mElemSize; }
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
    bool mIsFortranOrder;

    bool mHasDataOwnership;
};


typedef std::map<std::string, NpArray> NpArrayDict;
typedef std::pair<std::string, NpArray> NpArrayDictItem;


NpArrayDict npz_load(const std::string& fname);
NpArray npz_load(const std::string& fname, const std::string& varname);
NpArray npy_load(const std::string& fname);

void npy_save_data(const std::string& fname,
                   const unsigned char* data,const std::type_info& typeInfo,
                   const size_t elemSize, const size_t* shape, const size_t ndims,
                   const char mode='w');
void npz_save_data(const std::string& zipname, const std::string& name,
                   const unsigned char* data,const std::type_info& typeInfo,
                   const size_t elemSize, const size_t* shape, const size_t ndims,
                   const char mode='w');


template<typename T> void npy_save(std::string fname,
                                   const T* data, const size_t* shape, const size_t ndims,
                                   const char mode='w')
{
    npy_save_data(fname, reinterpret_cast<const unsigned char*>(data),
                  typeid(T), sizeof(T), shape, ndims, mode);
}

template<typename T> void npz_save(const std::string& zipname, const std::string& name,
                                   const T* data, const size_t* shape, const size_t ndims,
                                   const char mode='w')
{
    npz_save_data(zipname, name, reinterpret_cast<const unsigned char*>(data),
                  typeid(T), sizeof(T), shape, ndims, mode);
}

}

#endif

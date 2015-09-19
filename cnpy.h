//Copyright (C) 2011  Carl Rogers
//Released under MIT License
//license available in LICENSE file, or at http://www.opensource.org/licenses/mit-license.php

/*
 * The orignal cnpy library was modified by
 * Alessandro Bacchini, allebacco@gmail.com
 */

#ifndef LIBCNPY_H_
#define LIBCNPY_H_


#include <string>
#include <cstring>
#include <vector>
#include <type_traits>
#include <complex>
#include <map>
#include <algorithm>
#include <iostream>
#include <climits>


namespace cnpy
{

/**
 * @brief Supported datatypes.
 */
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

/**
 * @brief Conversion from C type to cnpy::Type enum.
 * @return The cnpy::Type that reflect the C type.
 *
 * Example:
 * @code{.cpp}
 * cnpy::Type the_type_for_int = cnpy::type<int>();
 * @endcode
 */
template<typename _Tp> static Type type()
{
    if(std::is_same<_Tp, char>::value)
    {
        if(std::numeric_limits<char>::is_signed)
            return Type::Int8;
        else
            return Type::Uint8;
    }
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

/**
 * @brief The NpArray class
 *
 * Container class for the data read from the `npy` and `npz` files.
 *
 * The NpArray usually deallocate its data when destroyed. However,
 * it is possible to move the responsibility of deleting data from
 * the class to the developer.
 *
 * @code{.cpp}
 * NpArray arr = npy_load("arr.npy");
 * float* data = arr.data(); // Take a pointer to the data
 * arr.revokeDataOwnership(); // Take the responsibility of deleting data
 * ...
 * delete[] data; // Delete the data
 * // Do not use NpArray::data() anymore because it points to
 * // an unallocated memory zone
 * @endcode
 *
 * @note
 * Do not use NpArray::data() if the data ownership has been revoked from the
 * NpArray class instance.
 *
 */
class NpArray
{
public:
    /**
     * @brief Constructor for an empty NpArray
     */
    NpArray() :
        mData(nullptr),
        mElemSize(0),
        mDataSize(0),
        mIsFortranOrder(false),
        mDtype(Type::Void),
        mHasDataOwnership(true)
    {}

    /**
     * @brief Constructor for a NpArray
     * @param shape Shape of the data
     * @param elSize Size of each element
     * @param isFortran true if the data is in fortran order (col-majour)
     * @param data The data to copy in the array
     *
     *
     */
    NpArray(const std::vector<size_t>& shape,
            const size_t elSize,
            const Type dataType,
            const bool isFortran=false,
            const unsigned char* data=nullptr) :
        mShape(shape),
        mElemSize(elSize),
        mIsFortranOrder(isFortran),
        mDtype(dataType),
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
        move(other);
    }

    NpArray& operator=(NpArray&& other)
    {
        move(other);
        return *this;
    }

    /**
     * @brief data
     * @return Pointer to the data read from the `npy` or `npz` file.
     * @throws std::runtime_error If the ownership of the data has been revoked from instance of NpArray
     */
    unsigned char* data() {
        if (mHasDataOwnership==false)
            throw std::runtime_error("The data ownership has been revoked from the NpArray instance");
        return mData;
    }

    /**
     * @brief data
     * @return Pointer to the data read from the `npy` or `npz` file.
     * @throws std::runtime_error If the ownership of the data has been revoked from instance of NpArray
     */
    const unsigned char* data() const {
        if (mHasDataOwnership==false)
            throw std::runtime_error("The data ownership has been revoked from the NpArray instance");
        return mData;
    }

    /**
     * @brief Size of the array in the dimension.
     * @param i Dimension
     * @return The size of the NpArray in the dimension `i`
     */
    size_t shape(const size_t i) const { return mShape[i]; }

    /**
     * @brief Number of dimensions
     * @return The Number of dimensions
     */
    size_t nDims() const { return mShape.size(); }

    /**
     * @brief Number of elements in the NpArray
     * @return The number of elements
     */
    size_t numElements() const { return mDataSize/mElemSize; }

    /**
     * @brief Size in byte of each element
     * @return The size in byte of each element
     */
    size_t elemSize() const { return mElemSize; }

    /**
     * @brief Size in byte of the data in the NpArray
     * @return The size in byte of the data
     */
    size_t size() const { return mDataSize; }

    /**
     * @brief Order of teh data in the NpArray
     * @return true if the data is in column-major order (FORTRAN), false if the
     *         data is in row-major order (C)
     */
    bool isFortranOrder() const { return mIsFortranOrder; }

    /**
     * @brief Data ownership
     * @return true if the NpArray instance has the responsibility of deallocate
     *         internal data when destructed
     */
    bool hasDataOwnership() const { return mHasDataOwnership; }

    /**
     * @brief Revoke the responsibility of deleting internal data from the
     *        NpArray instance
     */
    void revokeDataOwnership() { mHasDataOwnership = false; }

    /**
     * @brief Check if the NpArray instance is empty
     * @return true if the NpArray is empty
     */
    bool empty() const { return mData==nullptr; }

    Type dtype() const { return mDtype; }

private:

    void move(NpArray& other)
    {
        mData = other.mData;
        other.mData = nullptr;

        mShape = std::move(other.mShape);

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
    Type mDtype;

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

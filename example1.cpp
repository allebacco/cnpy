#include <complex>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <cassert>

#include "cnpy.h"

const int Nx = 128;
const int Ny = 64;
const int Nz = 32;

int main()
{
    //create random data
    std::complex<double>* data = new std::complex<double>[Nx*Ny*Nz];
    for(int i = 0;i < Nx*Ny*Nz;i++)
        data[i] = std::complex<double>(rand(),rand());

    //save it to file
    const std::vector<size_t> shape = {Nz, Ny, Nx};
    cnpy::npy_save("arr1.npy",data, shape, 'w');

    //load it into a new array
    cnpy::NpArray arr = cnpy::npy_load("arr1.npy");
    std::complex<double>* loaded_data = reinterpret_cast<std::complex<double>*>(arr.data());
    
    //make sure the loaded data matches the saved data
    assert(arr.elemSize() == sizeof(std::complex<double>));
    assert(arr.nDims() == 3 && arr.shape(0) == Nz && arr.shape(1) == Ny && arr.shape(2) == Nx);
    for(int i = 0; i < Nx*Ny*Nz;i++)
        if(data[i] != loaded_data[i])
            throw std::runtime_error("data[i] != loaded_data[i]");

    //append the same data to file
    //npy array on file now has shape (Nz+Nz,Ny,Nx)
    cnpy::npy_save("arr1.npy", data, shape, 'a');

    //now write to an npz file
    //non-array variables are treated as 1D arrays with 1 element
    double myVar1 = 1.2;
    char myVar2 = 'a';
    std::vector<size_t> shape2 = {1};
    cnpy::npz_save("out.npz","myVar1", &myVar1, shape2, 'w'); //"w" overwrites any existing file
    cnpy::npz_save("out.npz","myVar2", &myVar2, shape2, 'a'); //"a" appends to the file we created above
    cnpy::npz_save("out.npz","arr1", data, shape, 'a'); //"a" appends to the file we created above

    //load a single var from the npz file
    cnpy::NpArray arr2 = cnpy::npz_load("out.npz", "arr1");

    cnpy::NpArray npMyVar1 = cnpy::npz_load("out.npz", "myVar1");
    double* myVar1Data = reinterpret_cast<double*>(npMyVar1.data());
    if(npMyVar1.nDims() != 1 || npMyVar1.shape(0) != 1)
        throw std::runtime_error("npMyVar1.nDims() != 1 || npMyVar1.shape(0) != 1");
    if(myVar1Data[0] != myVar1)
        throw std::runtime_error("myVar1Data[0] != myVar1");
    if(npMyVar1.size()!=sizeof(double))
        throw std::runtime_error("npMyVar1.size()!=sizeof(double): ");

    //load the entire npz file
    cnpy::NpArrayDict my_npz = cnpy::npz_load("out.npz");

    //check that the loaded myVar1 matches myVar1
    cnpy::NpArray& arr_mv1 = my_npz["myVar1"];
    double* mv1 = reinterpret_cast<double*>(arr_mv1.data());
    if(arr_mv1.nDims() != 1 || arr_mv1.shape(0) != 1)
        throw std::runtime_error("arr_mv1.nDims() != 1 || arr_mv1.shape(0) != 1");
    if(mv1[0] != myVar1)
        throw std::runtime_error("mv1[0] != myVar1");

    return 0;
}

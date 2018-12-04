#pragma once

#include "MatrixDevice.hpp"

using spann::MatrixDevice;

class MatrixOperations
{
  public:
    __device__ __host__
    void HadamardProduct( MatrixDevice&, MatrixDevice& );
    __device__ __host__
    void Transpose( MatrixDevice& );
    __device__ __host__
    void Add( MatrixDevice&, MatrixDevice& );
    __device__ __host__
    void Subtract( MatrixDevice&, MatrixDevice& );
    __device__ __host__
    void Subtract( MatrixDevice&, float* );
    __device__ __host__
    void Subtract( MatrixDevice&, float*, MatrixDevice& );
    __device__ __host__
    void Apply( MatrixDevice&, float (*function)( float ) );
    __device__ __host__
    void Assign( MatrixDevice&, MatrixDevice& );
    __device__ __host__
    void Multiplication( MatrixDevice&, MatrixDevice&, MatrixDevice& );
    __device__ __host__
    void Multiplication( float*, MatrixDevice&, MatrixDevice& );
    __device__ __host__
    void Copy( MatrixDevice&, float* );
    __device__ __host__
    void Copy( MatrixDevice&, MatrixDevice& );
    __device__ __host__
    void Fill( MatrixDevice&, float );
    __device__ __host__
    void Resize( MatrixDevice&, int, int );
    __device__ __host__
    float Get( MatrixDevice&, int, int );
    __device__ __host__
    void Delete( MatrixDevice& );
};
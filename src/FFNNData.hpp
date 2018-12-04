#pragma once

#include "MatrixOperations.hpp"

namespace spann
{
  __device__ __host__
  float Sigmoid( float );

  struct DatasetDevice
  {
    float* input;
    float* output;
  };

  struct FFNNData
  {
    int  input_size;
    int  output_size;
    int  total_layers;
    int* layers_size;

    MatrixOperations operations;

    __device__ __host__
    void EvaluateInput( DatasetDevice&, MatrixDevice*, MatrixDevice*, float& );
  };
}
#pragma once

#include <FFNN.hpp>
#include "MatrixDevice.hpp"
#include "FFNNData.hpp"

namespace spann
{
  __global__
  void Kernel( int n, FFNNData* test, DatasetDevice* dataset, MatrixDevice* W, MatrixDevice* bias, float* error )
  {
    int start     = blockIdx.x * blockDim.x + threadIdx.x;
    int increment = blockDim.x * gridDim.x;

    for( int index = start; index < n; index += increment ) 
    {
      test->EvaluateInput( dataset[index], W, bias, error[index] );
    }
  }

  class FFNNHandler
  {
    public:
      int N;

      MatrixDevice* host_W  , * host_bias;
      MatrixDevice* bridge_W, * bridge_bias;
      MatrixDevice* device_W, * device_bias;

      float* host_error , * device_error ;
      
      DatasetDevice* host_dataset, * device_dataset, * bridge_dataset;

      FFNNData* device_ffnn;
      FFNNData* host_ffnn;

      FFNN* ffnn_original;

      FFNNHandler( int N )
      {
        int input  = 2;
        int output = 1;
        int hidden = 5;
        int layers = 3;

        host_dataset   = new DatasetDevice[N];
        bridge_dataset = new DatasetDevice[N];
        cudaMalloc( (void**)& device_dataset , N * sizeof( DatasetDevice ) );

        ffnn_original = new FFNN( input, layers, hidden, output, spann::sigmoid );

        ffnn_original->LoadOnesWeights( );

        this->N = N;

        host_W    = MatrixToDevice( ffnn_original->W   , ffnn_original->layers_n );
        host_bias = MatrixToDevice( ffnn_original->bias, ffnn_original->layers_n );

        device_W    = NewMatrixDevice( ffnn_original->layers_n );
        device_bias = NewMatrixDevice( ffnn_original->layers_n );

        bridge_W    = new MatrixDevice[ffnn_original->layers_n];
        bridge_bias = new MatrixDevice[ffnn_original->layers_n];

        InitMatrixDevice( device_W   , host_W   , ffnn_original->layers_n );
        InitMatrixDevice( device_bias, host_bias, ffnn_original->layers_n );

        CopyMatrixHostDevice( device_W   , bridge_W   , host_W   , ffnn_original->layers_n );
        CopyMatrixHostDevice( device_bias, bridge_bias, host_bias, ffnn_original->layers_n );

        host_ffnn   = new FFNNData;
        device_ffnn = NewFFNNDataDevice();

        InitFFNNDataHost( host_ffnn, input, output, hidden, layers );
        
        CopyFFNNDataHostDevice( device_ffnn, host_ffnn );
      }

      void Run()
      {
        SetupDataset();

        host_error = new float[N] { 0 };

        cudaMalloc( (void**)& device_error, N * sizeof( float ) );

        cudaMemcpy( device_error, host_error, N * sizeof( float ), cudaMemcpyHostToDevice );

        int block_size = 256;
        int num_blocks = ( N + block_size - 1 ) / block_size;

        Kernel<<<num_blocks, block_size>>>( N, device_ffnn, device_dataset, device_W, device_bias, device_error );

        cudaMemcpy( host_error, device_error, N * sizeof( float ), cudaMemcpyDeviceToHost );

        for( int i = 0; i < N; ++i )
        {
          std::cout << "Error[" << i << "] =  " << host_error[i] << std::endl;
        }
      }

      template<class T>
      MatrixDevice* MatrixToDevice( Matrix<T>* original, int elements )
      {
        MatrixDevice* device = new MatrixDevice[elements];
        for( int i = 0; i < elements; ++i )
        {
          device[i].rows    = original[i].rows;
          device[i].columns = original[i].columns;

          device[i].data = new float[ original[i].rows * original[i].columns ];
          device[i].tmp  = new float[ original[i].rows * original[i].columns ];
          
          for( int j = 0; j < original[i].rows * original[i].columns; ++j )
          {
            device[i].data[j] = original[i].data[j];
            device[i].tmp[j]  = 0;
          }
        }
        return device;
      }

      MatrixDevice* NewMatrixDevice( int elements )
      {
        MatrixDevice* result;
        cudaMalloc( (void**)& result, elements * sizeof( MatrixDevice ) );
        return result;
      }

      void InitMatrixDevice( MatrixDevice* device_matrix, MatrixDevice* host_matrix, int elements )
      {
        // Copying non-pointer data to device object
        cudaMemcpy( device_matrix, host_matrix, elements * sizeof( MatrixDevice ), cudaMemcpyHostToDevice );
      }

      void CopyMatrixHostDevice( MatrixDevice* device, MatrixDevice* bridge, MatrixDevice* host, int elements )
      {
        for( int i = 0; i < elements; ++i )
        {
          int array_size  = host[i].rows * host[i].columns;
          // Allocate device data   
          cudaMalloc( (void**)& bridge[i].data, array_size * sizeof( float ) );
          cudaMalloc( (void**)& bridge[i].tmp , array_size * sizeof( float ) );

          // Copy data from host to device
          cudaMemcpy( bridge[i].data, host[i].data, array_size * sizeof( float ), cudaMemcpyHostToDevice );
          cudaMemcpy( bridge[i].tmp , host[i].tmp , array_size * sizeof( float ), cudaMemcpyHostToDevice );

          // NOTE: Binding pointers with device
          cudaMemcpy( &( device[i].data ), &bridge[i].data, sizeof( device[i].data ), cudaMemcpyHostToDevice );
          cudaMemcpy( &( device[i].tmp ) , &bridge[i].tmp , sizeof( device[i].tmp ) , cudaMemcpyHostToDevice );
        }
      }

      void CopyMatrixDeviceHost( MatrixDevice* device, MatrixDevice* bridge, MatrixDevice* host, int elements )
      {
        for( int i = 0; i < elements; ++i )
        {
          int array_size  = host[i].rows * host[i].columns;

          cudaMemcpy( &bridge[i].data, &( device[i].data ), sizeof( bridge[i].data )    , cudaMemcpyHostToDevice );
          cudaMemcpy( host[i].data   , bridge[i].data     , array_size * sizeof( float ), cudaMemcpyDeviceToHost );
        }
      }

      void DeviceToHost()
      {
        CopyMatrixDeviceHost( device_W, bridge_W, host_W, ffnn_original->layers_n );
      }

      FFNNData* NewFFNNDataDevice()
      {
        FFNNData* result;
        cudaMalloc( (void**) &result, sizeof( FFNNData ) );
        return result;
      }

      void InitFFNNDataHost( FFNNData* host, int input, int output, int hidden, int layers )
      {
        host->input_size   = input;
        host->output_size  = output;
        host->total_layers = ffnn_original->layers_n;
        host->layers_size  = new int[host->total_layers];

        for( int i = 0; i < host->total_layers; ++i )
        {
          host->layers_size[i] = hidden;
        }
      }

      void CopyFFNNDataHostDevice( FFNNData* device, FFNNData* host )
      {
        cudaMemcpy( device, host, sizeof( FFNNData ), cudaMemcpyHostToDevice );
        int* layers_size;
        cudaMalloc( (void**)& layers_size, host->total_layers * sizeof( int ) );
        cudaMemcpy( layers_size, host->layers_size, host->total_layers * sizeof( int ), cudaMemcpyHostToDevice );
        cudaMemcpy( &( device->layers_size ), &layers_size, sizeof( device->layers_size ), cudaMemcpyHostToDevice );
      }

      void SetupDataset()
      {
      	for( int i = 0; i < N; ++i )
        {
          host_dataset[i].input  = new float[host_ffnn->input_size] { 1, 1 };
          host_dataset[i].output = new float[host_ffnn->output_size] { 0 };

          cudaMalloc( (void**)& bridge_dataset[i].input , host_ffnn->input_size * sizeof( float ) );
          cudaMalloc( (void**)& bridge_dataset[i].output, host_ffnn->output_size * sizeof( float ) );

          // Copy data from host to device
          cudaMemcpy( bridge_dataset[i].input , host_dataset[i].input , host_ffnn->input_size * sizeof( float ) , cudaMemcpyHostToDevice );
          cudaMemcpy( bridge_dataset[i].output, host_dataset[i].output, host_ffnn->output_size * sizeof( float ), cudaMemcpyHostToDevice );

          // NOTE: Binding pointers with device
          cudaMemcpy( &( device_dataset[i].input ) , &bridge_dataset[i].input , sizeof( device_dataset[i].input ) , cudaMemcpyHostToDevice );
          cudaMemcpy( &( device_dataset[i].output ), &bridge_dataset[i].output, sizeof( device_dataset[i].output ), cudaMemcpyHostToDevice );
        }
      }

      ~FFNNHandler( )
      {
        delete[] host_dataset;
        delete ffnn_original;
        delete host_ffnn;
        delete[] host_error;
        cudaFree( device_dataset );
        for( int i = 0; i < N; ++i )
        {
          cudaFree( bridge_dataset[i].input );
          cudaFree( bridge_dataset[i].output );
        }
        cudaFree( device_error );
      }
  };
}
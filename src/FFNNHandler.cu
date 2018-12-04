#include "FFNNHandler.hpp"

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

  FFNNHandler::FFNNHandler( int N, int input, int output, int hidden, int layers )
  {
    host_dataset   = new DatasetDevice[N];
    bridge_dataset = new DatasetDevice[N];
    cudaMalloc( (void**)& device_dataset , N * sizeof( DatasetDevice ) );

    std::cout << "ffnn_original = new FFNN( "
              << input << ", " << layers << ", " << hidden << ", " << output
              << ", spann::sigmoid );" << std::endl;

    ffnn_original = new FFNN( input, layers, hidden, output, spann::sigmoid );

    ffnn_original->LoadRandomWeights( 0 );

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

  void FFNNHandler::Run()
  {
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

  MatrixDevice* FFNNHandler::NewMatrixDevice( int elements )
  {
    MatrixDevice* result;
    cudaMalloc( (void**)& result, elements * sizeof( MatrixDevice ) );
    return result;
  }

  void FFNNHandler::InitMatrixDevice( MatrixDevice* device_matrix, MatrixDevice* host_matrix, int elements )
  {
    // Copying non-pointer data to device object
    cudaMemcpy( device_matrix, host_matrix, elements * sizeof( MatrixDevice ), cudaMemcpyHostToDevice );
  }

  void FFNNHandler::CopyMatrixHostDevice( MatrixDevice* device, MatrixDevice* bridge, MatrixDevice* host, int elements )
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

  void FFNNHandler::CopyMatrixDeviceHost( MatrixDevice* device, MatrixDevice* bridge, MatrixDevice* host, int elements )
  {
    for( int i = 0; i < elements; ++i )
    {
      int array_size  = host[i].rows * host[i].columns;

      cudaMemcpy( &bridge[i].data, &( device[i].data ), sizeof( bridge[i].data )    , cudaMemcpyHostToDevice );
      cudaMemcpy( host[i].data   , bridge[i].data     , array_size * sizeof( float ), cudaMemcpyDeviceToHost );
    }
  }

  void FFNNHandler::DeviceToHost()
  {
    CopyMatrixDeviceHost( device_W, bridge_W, host_W, ffnn_original->layers_n );
  }

  FFNNData* FFNNHandler::NewFFNNDataDevice()
  {
    FFNNData* result;
    cudaMalloc( (void**) &result, sizeof( FFNNData ) );
    return result;
  }

  void FFNNHandler::InitFFNNDataHost( FFNNData* host, int input, int output, int hidden, int layers )
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

  void FFNNHandler::CopyFFNNDataHostDevice( FFNNData* device, FFNNData* host )
  {
    cudaMemcpy( device, host, sizeof( FFNNData ), cudaMemcpyHostToDevice );
    int* layers_size;
    cudaMalloc( (void**)& layers_size, host->total_layers * sizeof( int ) );
    cudaMemcpy( layers_size, host->layers_size, host->total_layers * sizeof( int ), cudaMemcpyHostToDevice );
    cudaMemcpy( &( device->layers_size ), &layers_size, sizeof( device->layers_size ), cudaMemcpyHostToDevice );
  }

  void FFNNHandler::SetupDataset( Rcpp::NumericMatrix in_data, Rcpp::NumericMatrix out_data )
  {
    for( int i = 0; i < N; ++i )
    {
      host_dataset[i].input  = new float[host_ffnn->input_size];
      host_dataset[i].output = new float[host_ffnn->output_size];

      for( int j = 0; j < host_ffnn->input_size; j++ )
      {
        host_dataset[i].input[j] = in_data( i, j );
      }

      for( int j = 0; j < host_ffnn->output_size; j++ )
      {
        host_dataset[i].output[j] = out_data( i, j );
      }

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

  FFNNHandler::~FFNNHandler( )
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
}
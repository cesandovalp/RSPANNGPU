#include <cstdlib>
#include <iostream>
#include "MatrixOperations.hpp"
#include "FFNNHandler.hpp"

using namespace std;
using namespace spann;

void PrintMatrix( MatrixDevice& M )
{
  for( int i = 0; i < M.rows; ++i )
  {
    for( int j = 0; j < M.columns; ++j )
      std::cout << M[i][j] << "\t|";
    std::cout << std::endl;
  }
}

void EvaluateGPU( Rcpp::NumericMatrix in_data, Rcpp::NumericMatrix out_data )
{
  // N, input, output, hidden, layers )
  FFNNHandler ffnn_handler( in_data.nrow(), in_data.ncol(), out_data.ncol(), 150, 3 );

  ffnn_handler.SetupDataset( in_data, out_data );

  ffnn_handler.Run();

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  ffnn_handler.DeviceToHost();

  cout << "Dataset rows: " << in_data.nrow()  << endl;
  cout << "FFNN inputs: "  << in_data.ncol()  << endl;
  cout << "FFNN outputs: " << out_data.ncol() << endl;
} 

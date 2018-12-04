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

void hello( int N )
//int main( void )
{
  FFNNHandler ffnn_handler( N );

  //int N = 1 << 20;
  //int N = 2560;

  ffnn_handler.Run();

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  ffnn_handler.DeviceToHost();

  /*PrintMatrix( ffnn_handler.host_W[0] );
  PrintMatrix( ffnn_handler.host_bias[0] );*/

  cout << "CPU Hello World! " << N << endl;
} 

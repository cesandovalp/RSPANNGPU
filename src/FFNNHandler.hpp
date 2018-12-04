#pragma once

#include <FFNN.hpp>
#include "MatrixDevice.hpp"
#include "FFNNData.hpp"
#include <Rcpp.h>

namespace spann
{
  __global__
  void Kernel( int, FFNNData*, DatasetDevice*, MatrixDevice*, MatrixDevice*, float* );

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

      FFNNHandler( int, int, int, int, int );

      void Run();

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

      MatrixDevice* NewMatrixDevice( int );

      void InitMatrixDevice( MatrixDevice*, MatrixDevice*, int );

      void CopyMatrixHostDevice( MatrixDevice*, MatrixDevice*, MatrixDevice*, int );

      void CopyMatrixDeviceHost( MatrixDevice*, MatrixDevice*, MatrixDevice*, int );

      void DeviceToHost();

      FFNNData* NewFFNNDataDevice();

      void InitFFNNDataHost( FFNNData*, int, int, int, int );

      void CopyFFNNDataHostDevice( FFNNData*, FFNNData* );

      void SetupDataset( Rcpp::NumericMatrix, Rcpp::NumericMatrix );

      ~FFNNHandler( );
  };
}
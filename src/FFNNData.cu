#include "FFNNData.hpp"

namespace spann
{
  float Sigmoid( float x )
  {
    return 1 / ( 1 + exp( -x ) );
  }

  void FFNNData::EvaluateInput( DatasetDevice& dataset, MatrixDevice* W, MatrixDevice* bias, float& error )
  {
    int layer = 0;
    MatrixDevice X, Z, A;

    X.SetSize( 1, input_size, 0 );
    Z.SetSize( 1, W[layer].columns, 0 );
    A.SetSize( 1, W[layer].columns, 0 );

    operations.Copy( X, dataset.input );

    // Input
    // Z[l] = X * W[l] + b[l]
    operations.Multiplication( X, W[layer], Z );
    operations.Add( Z, bias[layer] );
    // a[l] = f( Z[l] )
    operations.Copy( A, Z );
    operations.Apply( A, Sigmoid );
    // Hidden
    for( ++layer ; layer < total_layers - 1; ++layer )
    {
      operations.Delete( Z );
      Z.SetSize( 1, W[layer].columns, 0 );
      operations.Fill( Z, 0.0 );
      // Z[l] = a[l - 1] * W[l] + b[l]
      operations.Multiplication( A, W[layer], Z );
      operations.Add( Z, bias[layer] );
      // a[l] = f( Z[l] )
      operations.Delete( A );
      A.SetSize( 1, W[layer].columns, 0 );
      operations.Copy( A, Z );
      operations.Apply( A, Sigmoid );
    }

    if( total_layers == 1 )
    {
      for( int i = 0; i < output_size; ++i )
        error += ( A.data[i] - dataset.output[i] ) * ( A.data[i] - dataset.output[i] );
      return;
    }

    operations.Delete( Z );
    Z.SetSize( 1, W[layer].columns, 0 );
    // Output
    operations.Multiplication( A, W[layer], Z );
    operations.Add( Z, bias[layer] );
    operations.Delete( A );
    A.SetSize( 1, W[layer].columns, 0 );
    operations.Copy( A, Z );
    operations.Apply( A, Sigmoid );

    for( int i = 0; i < output_size; ++i )
    {
      error += ( A.data[i] - dataset.output[i] ) * ( A.data[i] - dataset.output[i] );
    }
  }
}
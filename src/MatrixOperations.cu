#include "MatrixOperations.hpp"

void MatrixOperations::HadamardProduct( MatrixDevice& a, MatrixDevice& b )
{
  for( int i = 0; i < a.rows * a.columns; ++i )
    a.data[i] *= b.data[i];
}

void MatrixOperations::Transpose( MatrixDevice& matrix )
{
  int columns = matrix.columns;
  matrix.columns = matrix.rows;
  matrix.rows    = columns;

  for( int i = 0; i < matrix.columns; ++i )
    for( int j = 0; j < matrix.rows; ++j )
      ( matrix.data + ( i * matrix.columns ) )[j] = ( matrix.data + ( j * matrix.columns ) )[i];
}

void MatrixOperations::Apply( MatrixDevice& matrix, float (*function)( float ) )
{
  for( int i = 0; i < matrix.rows * matrix.columns; ++i )
    matrix.data[i] = function( matrix.data[i] );
}

void MatrixOperations::Add( MatrixDevice& a, MatrixDevice& b )
{
  for( int i = 0; i < a.rows * a.columns; ++i )
    a.data[i] += b.data[i];
}

void MatrixOperations::Subtract( MatrixDevice& a, MatrixDevice& b )
{
  for( int i = 0; i < a.rows * a.columns; ++i )
    a.data[i] -= b.data[i];
}

void MatrixOperations::Subtract( MatrixDevice& a, float* b )
{
  for( int i = 0; i < a.rows * a.columns; ++i )
    a.data[i] -= b[i];
}

void MatrixOperations::Assign( MatrixDevice& a, MatrixDevice& b )
{
  for( int i = 0; i < a.rows * a.columns; ++i )
    a.data[i] = b.data[i];
}

void MatrixOperations::Multiplication( MatrixDevice& a, MatrixDevice& b, MatrixDevice& result )
{
  for( int i = 0; i < a.rows; ++i )
    for( int j = 0; j < b.columns; ++j )
      for( int k = 0; k < a.columns; ++k )
        result.data[i * b.columns + j] += Get( a, i, k ) * Get( b, k, j );
}

void MatrixOperations::Multiplication( float* a, MatrixDevice& b, MatrixDevice& result )
{
  for( int i = 0; i < result.rows; ++i )
    for( int j = 0; j < b.columns; ++j )
      for( int k = 0; k < result.columns; ++k )
        result.data[i * b.columns + j] += a[i] * Get( b, k, j );
}

void MatrixOperations::Copy( MatrixDevice& matrix, float* array )
{
  for( int i = 0; i < matrix.rows * matrix.columns; ++i )
    matrix.data[i] = array[i];
}

void MatrixOperations::Copy( MatrixDevice& a, MatrixDevice& b )
{
  a.columns = b.columns;
  a.rows    = b.rows;

  for( int i = 0; i < a.rows * a.columns; ++i )
    a.data[i] = b.data[i];
}

void MatrixOperations::Fill( MatrixDevice& matrix, float value )
{
  for( int i = 0; i < matrix.rows * matrix.columns; ++i )
    matrix.data[i] = value;
}

void MatrixOperations::Resize( MatrixDevice& matrix, int rows, int columns )
{
  if( ( rows * columns ) == ( matrix.rows * matrix.columns ) )
  {
    matrix.rows    = rows;
    matrix.columns = columns;
  }
  else
  {
    matrix.data = new float[ rows * columns ];
  }
}

float MatrixOperations::Get( MatrixDevice& matrix, int row, int column )
{
  return *( matrix.data + ( row * matrix.columns ) + column );
}

void MatrixOperations::Delete( MatrixDevice& matrix )
{
  delete matrix.data;
  delete matrix.tmp;
}
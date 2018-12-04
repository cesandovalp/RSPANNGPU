#pragma once

#include <vector>
#include <iostream>

namespace spann
{
  class MatrixDevice
  {
    public:
      int rows, columns;
      float* data = 0;
      float* tmp  = 0;

      __host__ __device__
      MatrixDevice() { }

      __host__ __device__
      ~MatrixDevice()
      {
        delete[] data;
        delete[] tmp;
      }

      __host__ __device__
      void SetSize( int rows, int columns, float value = 0 )
      {
        this->rows    = rows;
        this->columns = columns;

        data = new float[rows * columns];
        tmp  = new float[rows * columns];

        for(int i = 0; i < rows * columns; ++i)
        {
          data[i] = value;
          tmp[i]  = value;
        }
      }

      __host__ __device__
      MatrixDevice( int rows, int columns ) : rows(rows), columns(columns)
      {
        data = new float[rows * columns];
        tmp  = new float[rows * columns];

        for( int i = 0; i < rows * columns; ++i )
        {
          data[i] = 0;
          tmp[i] = 0;
        }
      }

      MatrixDevice( const MatrixDevice& other )
      {
        data = new float[other.rows * other.columns];
        tmp  = new float[other.rows * other.columns];
        std::copy( other.data, other.data + ( other.rows * other.columns ), data );
        std::copy( other.tmp , other.tmp  + ( other.rows * other.columns ), tmp );

        rows    = other.rows;
        columns = other.columns;
      }

      MatrixDevice( const std::vector<float>& m, bool row = true )
      {
        if( row )
        {
          columns = 1;
          rows    = m.size();
        }
        else
        {
          columns = m.size();
          rows    = 1;
        }
        data = new float[m.size()];
        tmp  = new float[m.size()];
        for( int i = 0; i < m.size(); ++i )
        {
          data[i] = m[i];
          tmp[i]  = 0;
        }
      }

      float* operator[]( int row ) const { return data + ( row * columns ); }
      float* operator()( int index ) const { return data + index; }

      MatrixDevice& operator+=( const MatrixDevice& b )
      {
        for( int i = 0; i < rows * columns; ++i )
          data[i] += *b( i );
        return *this;
      }

      MatrixDevice operator+( const MatrixDevice& b )
      {
        MatrixDevice result = *this;
        result += b;
        return result;
      }

      MatrixDevice& operator-=( const MatrixDevice& b )
      {
        for( int i = 0; i < rows * columns; ++i )
          data[i] -= *b(i);
        return *this;
      }

      MatrixDevice operator-( const MatrixDevice& b )
      {
        MatrixDevice result = *this;
        result -= b;
        return result;
      }

      MatrixDevice& operator-=( const std::vector<float>& b )
      {
        for( int i = 0; i < rows * columns; ++i )
          data[i] -= b[i];
        return *this;
      }

      MatrixDevice& operator*=( const MatrixDevice& b )
      {
        *this = *this * b;
        return *this;
      }

      MatrixDevice& operator*=( const std::vector<float>& b )
      {
        *this = *this * b;
        return *this;
      }

      MatrixDevice& operator*=( const float& a )
      {
        *this = *this * a;
        return *this;
      }

      MatrixDevice operator*( const MatrixDevice& b )
      {
        MatrixDevice result( rows, b.columns );

        for( int i = 0; i < rows; ++i )
          for( int j = 0; j < b.columns; ++j )
            for( int k = 0; k < columns; ++k )
              result[i][j] += ( data + ( i * columns ) )[k] * b[k][j];

        return result;
      }

      MatrixDevice operator*( const std::vector<float>& b )
      {
        MatrixDevice result( rows, b.size() );

        for( int i = 0; i < rows; ++i )
          for( int j = 0; j < b.size(); ++j )
            for( int k = 0; k < columns; ++k )
              result[i][j] += ( data + ( i * columns ) )[k] * b[j];

        return result;
      }

      MatrixDevice operator*( const float& a )
      {
        MatrixDevice result( rows, columns );

        for( int i = 0; i < rows; ++i )
            for( int j = 0; j < columns; ++j )
              result[i][j] += ( data + ( i * columns ) )[j] * a;

        return result;
      }

      // Use only when dim(A*B) = dim(A)
      void Multiplication( const MatrixDevice& b )
      {
        float* swap;
        for( int i = 0; i < rows; ++i )
          for( int j = 0; j < b.columns; ++j )
            for( int k = 0; k < columns; ++k )
              tmp[i * columns + j] += data[i * columns + k] * b[k][j];
        swap = data;
        data = tmp;
        tmp  = swap;
        for( int i = 0; i < rows * columns; ++i )
          tmp[i] = 0;
      }

      void HadamardProduct( const MatrixDevice& b )
      {
        for( int i = 0; i < rows * columns; ++i )
          data[i] *= *b( i );
      }

      void Multiplication( const MatrixDevice& a, const MatrixDevice& b )
      {
        for( int i = 0; i < a.rows; ++i )
          for( int j = 0; j < b.columns; ++j )
            for( int k = 0; k < a.columns; ++k )
              data[i * b.columns + j] += a[i][k] * b[k][j];
      }

      MatrixDevice operator-() const
      {
        MatrixDevice result( rows, columns );
        for( int i = 0; i < rows * columns; ++i )
          *result( i ) = -data[i];
        return result;
      }

      MatrixDevice operator!() const
      {
        MatrixDevice result( columns, rows );
        for( int i = 0; i < columns; ++i )
          for( int j = 0; j < rows; ++j )
            result[i][j] = ( data + ( j * columns ) )[i];
        return result;
      }

      MatrixDevice& operator=( const MatrixDevice& other )
      {
        delete[] data;
        delete[] tmp;

        data = new float[other.rows * other.columns];
        tmp  = new float[other.rows * other.columns];
        std::copy( other.data, other.data + ( other.rows * other.columns ), data );
        std::copy( other.tmp , other.tmp  + ( other.rows * other.columns ), tmp );

        rows    = other.rows;
        columns = other.columns;

        return *this;
      }

      MatrixDevice& operator=( const float& other )
      {
        for( int i = 0; i < rows * columns; ++i )
          data[i] = other;

        return *this;
      }

      void Copy( const std::vector<float>& m )
      {
        columns = m.size();
        rows    = 1;

        for( int i = 0; i < m.size(); ++i )
        {
          data[i] = m[i];
          tmp[i] = 0;
        }
      }

      void Copy( float* m )
      {
        for( int i = 0; i < rows*columns; ++i )
        {
          data[i] = m[i];
          tmp[i] = 0;
        }
      }

      void Copy( const MatrixDevice& other )
      {
        columns = other.columns;
        rows    = other.rows;

        for( int i = 0; i < rows * columns; ++i )
          data[i] = *other( i );
      }

      void operator()( const MatrixDevice& other )
      {
        std::copy( other.data, other.data + ( other.rows * other.columns ), data );
      }

      MatrixDevice& operator()( float (*f)( float ) )
      {
        for(int i = 0; i < rows * columns; ++i)
          data[i] = f( data[i] );

        return *this;
      }

      MatrixDevice Apply( float (*f)( float ) )
      {
        MatrixDevice result = *this;
        result(f);

        return result;
      }

      std::vector<float> ToVector()
      {
        return std::vector<float>( data, data + rows * columns );
      }

      void ToVector( std::vector<float>& t )
      {
        t.assign( data, data + (rows * columns) );
      }

      void Initialize( int rows, int columns )
      {
        data = new float[rows * columns];
        tmp  = new float[rows * columns];

        this->rows    = rows;
        this->columns = columns;
      }

      float SumElements()
      {
        float result = 0;
        for( int i = 0; i < rows * columns; ++i )
          result += data[i];
        return result;
      }

      friend std::ostream& operator<< ( std::ostream& stream, const MatrixDevice& m )
      {
        for( int j = 0; j < m.rows; ++j )
        {
          for( int k = 0; k < m.columns; ++k )
            stream << m.data[ j * m.columns + k ] << '|';
          stream << '\n';
        }
        return stream;
      }
  };
}

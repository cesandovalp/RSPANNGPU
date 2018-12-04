#include <Rcpp.h>
using namespace Rcpp;

void EvaluateGPU( Rcpp::NumericMatrix, Rcpp::NumericMatrix );

// [[Rcpp::export]]
List EvaluateDataset( Rcpp::NumericMatrix in_data, Rcpp::NumericMatrix out_data )
{
    EvaluateGPU( in_data, out_data );

    CharacterVector x = CharacterVector::create( "foo", "bar" )  ;
    NumericVector y   = NumericVector::create( 0.0, 1.0 ) ;
    List z            = List::create( x, y ) ;

    return z ;
}

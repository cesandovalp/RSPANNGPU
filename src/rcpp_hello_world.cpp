#include <Rcpp.h>
using namespace Rcpp;

void hello(int);

// [[Rcpp::export]]
List rcpp_hello_world()
{
    //hello( 2560 );
    hello( 2560 );

    CharacterVector x = CharacterVector::create( "foo", "bar" )  ;
    NumericVector y   = NumericVector::create( 0.0, 1.0 ) ;
    List z            = List::create( x, y ) ;

    return z ;
}

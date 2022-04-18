#include "defines.h"
#include "diff.h"
#include "ncsr.h"

#include <cstdio>
#include <cstring>

#ifndef GT
        #define GT( t ) { clock_gettime( CLOCK_MONOTONIC, &t ); }
#endif

static double measure_time( const struct timespec t2, const struct timespec t1 )
{
        double t = (double) ( t2.tv_sec - t1.tv_sec ) + ( (double) ( t2.tv_nsec - t1.tv_nsec ) ) * 1e-9;
        return( t );
}

str_res test_ncsr( const UIN ompNT, const str_matCSR matCSR, const FPT * vec, const FPT * ref )
{
       // timed iterations
       double ti = 0.0, tt = 0.0;
       struct timespec t1, t2;
       FPT * res = (FPT *) calloc( matCSR.nrows, sizeof(FPT) ); TEST_POINTER( res );
       UIN i;
       for ( i = 0; i < NUM_ITE; i++ )
       {
               GT( t1 );
               ncsr( ompNT, matCSR, vec, res );
               GT( t2 );
               ti = measure_time( t2, t1 );
               tt = tt + ti;
       }
       // store results
       str_res sr;
       strcpy( sr.name, "ncsr" );
       sr.et    = tt / (double) NUM_ITE;
       sr.ot    = 0.0;
       sr.flops = ( 2.0 * ( (double) matCSR.nnz ) ) / sr.et;
       get_errors( matCSR.nrows, ref, res, &(sr.sErr) );
       free( res );
       return( sr );
}


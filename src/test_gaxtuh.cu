#include "defines.h"
#include "diff.h"
#include "gaxtuh.h"

#include <cstdio>
#include <cstring>

str_res test_gaxtuh( const UIN tpw, const UIN cbs, const str_matAXT matAXT, const FPT * ref )
{
	                                                                     // tpw - tiles per warp
	const UIN th           = matAXT.tileH;                               // tile's height
	const UIN thw          = matAXT.tileHW;                              // tile's half width
	const UIN tn           = ( ( matAXT.tileN + tpw - 1 ) / tpw ) * tpw; // number of tiles rounded to a multiple of tpw
	const UIN wn           = tn / tpw;                                   // number of warps needed
	const UIN wpb          = cbs / 32;                                   // warps per cuda block
	const UIN cbn          = ( wn + wpb - 1 ) / wpb;                     // number of cuda blocks needed
	const UIN devLenAX     = cbn * wpb * tpw * 2 * th * thw;             // lenAX for device
	const UIN devLenSEC    = cbn * wpb * tpw * thw;                      // lenSEC for device
//printf( "tpw      : %d\n", tpw           );
//printf( "th       : %d\n", th            );
//printf( "thw      : %d\n", thw           );
//printf( "tn       : %d\n", tn            );
//printf( "wn       : %d\n", wn            );
//printf( "wpb      : %d\n", wpb           );
//printf( "cbn      : %d\n", cbn           );
//printf( "lenAX    : %d\n", matAXT.lenAX  );
//printf( "devLenAX : %d\n", devLenAX      );
//printf( "lenSEC   : %d\n", matAXT.lenSEC );
//printf( "devLenSEC: %d\n", devLenSEC     );
	// allocate memory on GPU
	FPT * d_ax;  HANDLE_CUDA_ERROR( cudaMalloc( &d_ax,  devLenAX     * sizeof(FPT) ) ); TEST_POINTER( d_ax    );
	UIN * d_rwp; HANDLE_CUDA_ERROR( cudaMalloc( &d_rwp, devLenSEC    * sizeof(UIN) ) ); TEST_POINTER( d_rwp   );
	FPT * d_res; HANDLE_CUDA_ERROR( cudaMalloc( &d_res, matAXT.nrows * sizeof(FPT) ) ); TEST_POINTER( d_res   );
	HANDLE_CUDA_ERROR( cudaMemset( d_ax,  0, devLenAX  * sizeof(FPT) ) );
	HANDLE_CUDA_ERROR( cudaMemset( d_rwp, 0, devLenSEC * sizeof(UIN) ) );
	// copy necessary arrays to device
	HANDLE_CUDA_ERROR( cudaMemcpy( d_ax,  matAXT.ax,  matAXT.lenAX  * sizeof(FPT), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_rwp, matAXT.sec, matAXT.lenSEC * sizeof(UIN), cudaMemcpyHostToDevice ) );
	// create events for time measuring
	cudaEvent_t cet1; HANDLE_CUDA_ERROR( cudaEventCreate( &cet1 ) );
	cudaEvent_t cet2; HANDLE_CUDA_ERROR( cudaEventCreate( &cet2 ) );
	// timed iterations
	float ti = 0.0f, tt = 0.0f;
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		HANDLE_CUDA_ERROR( cudaMemset( d_res, 0, matAXT.nrows  * sizeof(FPT) ) );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet1 ) );
		gaxtuh <<<cbn, cbs>>> ( tpw, tn, th, d_ax, d_rwp, d_res );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventSynchronize( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventElapsedTime( &ti, cet1, cet2 ) );
		tt = tt + ti;
	}
	// destroy events for time measuring
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet1 ) );
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet2 ) );
	// copy result from device
	FPT * res = (FPT *) malloc( matAXT.nrows * sizeof(FPT) ); TEST_POINTER( res );
	HANDLE_CUDA_ERROR( cudaMemcpy( res, d_res, matAXT.nrows * sizeof(FPT), cudaMemcpyDeviceToHost ) );
	// free device memory
	HANDLE_CUDA_ERROR( cudaFree( d_ax    ) );
	HANDLE_CUDA_ERROR( cudaFree( d_rwp   ) );
	HANDLE_CUDA_ERROR( cudaFree( d_res   ) );
	// store results
	char TH[5];  sprintf( TH,  "%d", th );
	char TPW[5]; sprintf( TPW, "%d", tpw );
	char buffer[48];
	strcpy( buffer, "gaxtuh" );
	strcat( buffer, TH );
	strcat( buffer, "tpw" );
	strcat( buffer, TPW );
	str_res sr;
	strcpy( sr.name, buffer );
	sr.et    = ( (double) tt / (double) NUM_ITE ) * 1e-3;
	sr.ot    = 0.0;
	sr.flops = ( 2.0 * ( (double) matAXT.nnz ) ) / sr.et;
	get_errors( matAXT.nrows, ref, res, &(sr.sErr) );
	// free cpu memory
	free( res );
	return( sr );
}


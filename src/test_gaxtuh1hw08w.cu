#include "defines.h"
#include "diff.h"
#include "gaxtuh1hw08w.h"

#include <cstdio>
#include <cstring>

str_res test_gaxtuh1hw08w( const UIN wpw, const UIN cbs, const str_matAXT matAXT, const FPT * ref )
{
	                                                                     // wpw - workload per warp
	const UIN thw          = matAXT.tileHW;                              // tile half width
	const UIN tpw          = wpw / (2 * thw);                            // tiles per warp
	const UIN tn           = ( ( matAXT.tileN + tpw - 1 ) / tpw ) * tpw; // number of tiles rounded to a multiple of tpw
	const UIN wn           = tn / tpw;                                   // number of warps needed
	const UIN wpb          = cbs / 32;                                   // warps per cuda block
	const UIN cbn          = ( wn + wpb - 1 ) / wpb;                     // number of cuda blocks needed
	const UIN devLenAX     = cbn * wpb * tpw * 2 * thw;                  // lenAX for device
	const UIN devLenSEC    = cbn * wpb * tpw;                            // lenSEC for device
	// allocate memory on GPU
	FPT * d_ax;  HANDLE_CUDA_ERROR( cudaMalloc( &d_ax,    devLenAX     * sizeof(FPT) ) ); TEST_POINTER( d_ax    );
	UIN * d_rwp; HANDLE_CUDA_ERROR( cudaMalloc( &d_rwp,   devLenSEC    * sizeof(UIN) ) ); TEST_POINTER( d_rwp   );
	FPT * d_res; HANDLE_CUDA_ERROR( cudaMalloc( &d_res,   matAXT.nrows * sizeof(FPT) ) ); TEST_POINTER( d_res   );
	HANDLE_CUDA_ERROR( cudaMemset( d_ax,  0, devLenAX  * sizeof(FPT) ) );
	HANDLE_CUDA_ERROR( cudaMemset( d_rwp, 0, devLenSEC * sizeof(UIN) ) );
	// copy necessary arrays to device
	HANDLE_CUDA_ERROR( cudaMemcpy( d_ax,    matAXT.ax,    matAXT.lenAX  * sizeof(FPT), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_rwp,   matAXT.sec,   matAXT.lenSEC * sizeof(UIN), cudaMemcpyHostToDevice ) );
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
		gaxtuh1hw08w <<<cbn, cbs>>> ( tpw, tn, d_ax, d_rwp, d_res );
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
	char WPW[6]; sprintf( WPW, "%d", wpw );
	char buffer[48];
	strcpy( buffer, "gaxtuh1hw08w" );
	strcat( buffer, WPW );
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


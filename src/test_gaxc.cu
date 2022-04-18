#include "defines.h"
#include "diff.h"
#include "gaxc.h"
#include "axc.h"

#include <cstdio>
#include <cstring>

str_res test_gaxc( const UIN cudaBlockSize, const str_matAXC matAXC, const FPT * ref )
{
	// 
	const UIN cudaBlockNum = ( (matAXC.nrows * 32) + cudaBlockSize - 1 ) / cudaBlockSize;
	// allocate memory on GPU
	FPT * d_ax;    HANDLE_CUDA_ERROR( cudaMalloc( &d_ax,    matAXC.lenAX                * sizeof(FPT) ) ); TEST_POINTER( d_ax    );
	UIN * d_brp;   HANDLE_CUDA_ERROR( cudaMalloc( &d_brp,   matAXC.lenBRP               * sizeof(UIN) ) ); TEST_POINTER( d_brp   );
	FPT * d_res;   HANDLE_CUDA_ERROR( cudaMalloc( &d_res,   matAXC.nrows                * sizeof(FPT) ) ); TEST_POINTER( d_res   );
	// copy necessary arrays to device
	HANDLE_CUDA_ERROR( cudaMemset( d_ax,  0, matAXC.lenAX  * sizeof(FPT) ) );
	HANDLE_CUDA_ERROR( cudaMemset( d_brp, 0, matAXC.lenBRP * sizeof(UIN) ) );
	HANDLE_CUDA_ERROR( cudaMemset( d_res, 0, matAXC.nrows  * sizeof(FPT) ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_ax,    matAXC.ax,    matAXC.lenAX  * sizeof(FPT), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_brp,   matAXC.brp,   matAXC.lenBRP * sizeof(UIN), cudaMemcpyHostToDevice ) );
	// create events for time measuring
	cudaEvent_t cet1; HANDLE_CUDA_ERROR( cudaEventCreate( &cet1 ) );
	cudaEvent_t cet2; HANDLE_CUDA_ERROR( cudaEventCreate( &cet2 ) );
	// timed iterations
	float ti = 0.0f, tt = 0.0f;
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		HANDLE_CUDA_ERROR( cudaEventRecord( cet1 ) );
		gaxc <<<cudaBlockNum, cudaBlockSize>>> ( matAXC.nrows, d_ax, d_brp, d_res );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventSynchronize( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventElapsedTime( &ti, cet1, cet2 ) );
		tt = tt + ti;
	}
	// destroy events for time measuring
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet1 ) );
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet2 ) );
	// copy result from device
	FPT * res = (FPT *) malloc( matAXC.nrows * sizeof(FPT) ); TEST_POINTER( res );
	HANDLE_CUDA_ERROR( cudaMemcpy( res, d_res, matAXC.nrows * sizeof(FPT), cudaMemcpyDeviceToHost ) );
	// free device memory
	HANDLE_CUDA_ERROR( cudaFree( d_ax    ) );
	HANDLE_CUDA_ERROR( cudaFree( d_brp   ) );
	HANDLE_CUDA_ERROR( cudaFree( d_res   ) );
	// store results
	str_res sr;
	strcpy( sr.name, "gaxc" );
	sr.et    = ( (double) tt / (double) NUM_ITE ) * 1e-3;
	sr.ot    = 0.0;
	sr.flops = ( 2.0 * ( (double) matAXC.nnz ) ) / sr.et;
	get_errors( matAXC.nrows, ref, res, &(sr.sErr) );
	// free cpu memory
	free( res );
	return( sr );
}


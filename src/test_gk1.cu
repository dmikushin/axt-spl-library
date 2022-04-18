#include "defines.h"
#include "diff.h"
#include "gk1.h"
#include "k1.h"

#include <cstdio>
#include <cstring>

str_res test_gk1( const UIN cudaBlockSize, const str_matK1 matK1, const FPT * vec, const FPT * ref )
{
	// 
	UIN cudaBlockNum = ( matK1.nrows + cudaBlockSize - 1 ) / cudaBlockSize;
	// allocate memory on GPU
	FPT * d_val;   HANDLE_CUDA_ERROR( cudaMalloc( &d_val,   matK1.lenVC                 * sizeof(FPT) ) ); TEST_POINTER( d_val   );
	UIN * d_col;   HANDLE_CUDA_ERROR( cudaMalloc( &d_col,   matK1.lenVC                 * sizeof(UIN) ) ); TEST_POINTER( d_col   );
	UIN * d_nmc;   HANDLE_CUDA_ERROR( cudaMalloc( &d_nmc,   matK1.chunkNum              * sizeof(UIN) ) ); TEST_POINTER( d_nmc   );
	UIN * d_chp;   HANDLE_CUDA_ERROR( cudaMalloc( &d_chp,   matK1.chunkNum              * sizeof(UIN) ) ); TEST_POINTER( d_chp   );
	UIN * d_permi; HANDLE_CUDA_ERROR( cudaMalloc( &d_permi, matK1.chunkNum * CHUNK_SIZE * sizeof(UIN) ) ); TEST_POINTER( d_permi );
	FPT * d_vec;   HANDLE_CUDA_ERROR( cudaMalloc( &d_vec,   matK1.nrows                 * sizeof(FPT) ) ); TEST_POINTER( d_vec   );
	FPT * d_res;   HANDLE_CUDA_ERROR( cudaMalloc( &d_res,   matK1.nrows                 * sizeof(FPT) ) ); TEST_POINTER( d_res   );
	// copy necessary arrays to device
	HANDLE_CUDA_ERROR( cudaMemcpy( d_val,   matK1.val,   matK1.lenVC                 * sizeof(FPT), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_col,   matK1.col,   matK1.lenVC                 * sizeof(UIN), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_nmc,   matK1.nmc,   matK1.chunkNum              * sizeof(UIN), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_chp,   matK1.chp,   matK1.chunkNum              * sizeof(UIN), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_permi, matK1.permi, matK1.chunkNum * CHUNK_SIZE * sizeof(UIN), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_vec,   vec,         matK1.nrows                 * sizeof(FPT), cudaMemcpyHostToDevice ) );
	// create events for time measuring
	cudaEvent_t cet1; HANDLE_CUDA_ERROR( cudaEventCreate( &cet1 ) );
	cudaEvent_t cet2; HANDLE_CUDA_ERROR( cudaEventCreate( &cet2 ) );
	// timed iterations
	float ti = 0.0f, tt = 0.0f;
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		HANDLE_CUDA_ERROR( cudaEventRecord( cet1 ) );
		gk1 <<<cudaBlockNum, cudaBlockSize>>> (  matK1.nrows, d_val, d_col, d_nmc, d_chp, d_permi, d_vec, d_res );
		HANDLE_CUDA_ERROR( cudaEventRecord( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventSynchronize( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventElapsedTime( &ti, cet1, cet2 ) );
		tt = tt + ti;
	}
	// destroy events for time measuring
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet1 ) );
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet2 ) );
	// copy result from device
	FPT * res = (FPT *) malloc( matK1.nrows * sizeof(FPT) ); TEST_POINTER( res );
	HANDLE_CUDA_ERROR( cudaMemcpy( res, d_res, matK1.nrows * sizeof(FPT), cudaMemcpyDeviceToHost ) );
	// free device memory
	HANDLE_CUDA_ERROR( cudaFree( d_val   ) );
	HANDLE_CUDA_ERROR( cudaFree( d_col   ) );
	HANDLE_CUDA_ERROR( cudaFree( d_nmc   ) );
	HANDLE_CUDA_ERROR( cudaFree( d_chp   ) );
	HANDLE_CUDA_ERROR( cudaFree( d_permi ) );
	HANDLE_CUDA_ERROR( cudaFree( d_vec   ) );
	HANDLE_CUDA_ERROR( cudaFree( d_res   ) );
	// store results
	str_res sr;
	strcpy( sr.name, "gk1" );
	sr.et    = ( (double) tt / (double) NUM_ITE ) * 1e-3;
	sr.ot    = 0.0;
	sr.flops = ( 2.0 * ( (double) matK1.nnz ) ) / sr.et;
	get_errors( matK1.nrows, ref, res, &(sr.sErr) );
	// free cpu memory
	free( res );
	return( sr );
}


#include "defines.h"
#include "diff.h"
#include "gaxtch.h"

#include <cstdio>
#include <cstring>

str_res test_gaxtch( const UIN cudaBlockSize, const str_matAXT matAXT, const FPT * ref )
{
	const UIN th           = matAXT.tileH;
	char TH[5]; sprintf( TH, "%d", th );
	char buffer[48];
	strcpy( buffer, "gaxtch" );
	strcat( buffer, TH );
	str_res sr;
	strcpy( sr.name, buffer );
	if ( ( (strcmp(matAXT.name, "M24_circuit5M.bin") == 0) && (cudaBlockSize == 1024) ) || ( (strcmp(matAXT.name, "M23_delaunay_n23.bin") == 0) && (cudaBlockSize == 1024) ) )
	{
		sr.et        = 0.0;
		sr.ot        = 0.0;
		sr.flops     = 0.0;
		sr.sErr.aErr = 0.0;
		sr.sErr.rErr = 0.0;
		sr.sErr.pos  = 0;
	}
	else
	{
		// 
		const UIN tn           = matAXT.tileN;
		const UIN thw          = matAXT.tileHW;
		const UIN log          = matAXT.log;
		const UIN cudaBlockNum = ( (tn*32) + cudaBlockSize - 1 ) / cudaBlockSize;
		const UIN wpb          = cudaBlockSize / 32;
		const UIN devLenAX     = cudaBlockNum * 2 * th * thw * wpb;
		const UIN devLenSEC    = cudaBlockNum     * th * thw * wpb;
		// allocate memory on GPU
		FPT * d_ax;  HANDLE_CUDA_ERROR( cudaMalloc( &d_ax,  devLenAX      * sizeof(FPT) ) ); TEST_POINTER( d_ax  );
		UIN * d_hdr; HANDLE_CUDA_ERROR( cudaMalloc( &d_hdr, devLenSEC     * sizeof(UIN) ) ); TEST_POINTER( d_hdr );
		FPT * d_res; HANDLE_CUDA_ERROR( cudaMalloc( &d_res, matAXT.nrows  * sizeof(FPT) ) ); TEST_POINTER( d_res );
		// copy necessary arrays to device
		HANDLE_CUDA_ERROR( cudaMemset( d_ax,  0, devLenAX  * sizeof(FPT) ) );
		HANDLE_CUDA_ERROR( cudaMemset( d_hdr, 0, devLenSEC * sizeof(UIN) ) );
		HANDLE_CUDA_ERROR( cudaMemcpy( d_ax,  matAXT.ax,  matAXT.lenAX  * sizeof(FPT), cudaMemcpyHostToDevice ) );
		HANDLE_CUDA_ERROR( cudaMemcpy( d_hdr, matAXT.sec, matAXT.lenSEC * sizeof(UIN), cudaMemcpyHostToDevice ) );
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
			gaxtch <<<cudaBlockNum, cudaBlockSize>>> ( log, th, d_ax, d_hdr, d_res );
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
		HANDLE_CUDA_ERROR( cudaFree( d_ax  ) );
		HANDLE_CUDA_ERROR( cudaFree( d_hdr ) );
		HANDLE_CUDA_ERROR( cudaFree( d_res ) );
		// store results
		sr.et    = ( (double) tt / (double) NUM_ITE ) * 1e-3;
		sr.ot    = 0.0;
		sr.flops = ( 2.0 * ( (double) matAXT.nnz ) ) / sr.et;
		get_errors( matAXT.nrows, ref, res, &(sr.sErr) );
		// free cpu memory
		free( res );
	}
	return( sr );
}


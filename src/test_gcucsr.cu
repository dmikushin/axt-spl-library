#include "defines.h"
#include "diff.h"

#include <cstdio>
#include <cstring>
#include <cusparse.h>

#define HANDLE_CUSPARSE_ERROR( cseID ) { if ( cseID != CUSPARSE_STATUS_SUCCESS ) { printf( "FILE: %s LINE: %d CUBLAS_ERROR: %s\n", __FILE__, __LINE__, cusparseGetErrorMessage( cseID ) ); printf( "\nvim %s +%d\n", __FILE__, __LINE__); exit( EXIT_FAILURE ); } }

static __host__ const char * cusparseGetErrorMessage( cusparseStatus_t statusID )
{
        switch(statusID)
        {
                case CUSPARSE_STATUS_NOT_INITIALIZED:           return "CUSPARSE_STATUS_NOT_INITIALIZED";
                case CUSPARSE_STATUS_ALLOC_FAILED:              return "CUSPARSE_STATUS_ALLOC_FAILED";
                case CUSPARSE_STATUS_INVALID_VALUE:             return "CUSPARSE_STATUS_INVALID_VALUE";
                case CUSPARSE_STATUS_ARCH_MISMATCH:             return "CUSPARSE_STATUS_ARCH_MISMATCH";
                case CUSPARSE_STATUS_MAPPING_ERROR:             return "CUSPARSE_STATUS_MAPPING_ERROR";
                case CUSPARSE_STATUS_EXECUTION_FAILED:          return "CUSPARSE_STATUS_EXECUTION_FAILED";
                case CUSPARSE_STATUS_INTERNAL_ERROR:            return "CUSPARSE_STATUS_INTERNAL_ERROR";
                case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
        }
        return "<cusparse unknown>";
}

str_res test_gcucsr( const str_matCSR matCSR, const FPT * vec, const FPT * ref )
{
	// get parameteres for cuSPARSE
	const UIN                     nrows = matCSR.nrows;
	const UIN                       nnz = matCSR.nnz;
	cusparseHandle_t    cusparseH = NULL;
	const cusparseSpMVAlg_t cusparseAM = CUSPARSE_SPMV_CSR_ALG1;
	const cusparseOperation_t cusparseO = CUSPARSE_OPERATION_NON_TRANSPOSE;
	cusparseSpMatDescr_t cusparseMD = NULL;
	cusparseDnVecDescr_t cusparseVD1 = NULL, cusparseVD2 = NULL;
	size_t    cudaSpaceBufferSize;
	const FPT                      zero = (FPT)  0;
	const FPT                       one = (FPT)  1;
	#if FP_TYPE == FP_FLOAT
		cudaDataType cudaDT = CUDA_R_32F;
	#else
		cudaDataType cudaDT = CUDA_R_64F;
	#endif
	// allocate memory on GPU
	FPT * d_val; HANDLE_CUDA_ERROR( cudaMalloc( &d_val,           nnz * sizeof(FPT) ) ); TEST_POINTER( d_val );
	int * d_col; HANDLE_CUDA_ERROR( cudaMalloc( &d_col,           nnz * sizeof(int) ) ); TEST_POINTER( d_col );
	int * d_row; HANDLE_CUDA_ERROR( cudaMalloc( &d_row, ( nrows + 1 ) * sizeof(int) ) ); TEST_POINTER( d_row );
	FPT * d_vec; HANDLE_CUDA_ERROR( cudaMalloc( &d_vec,         nrows * sizeof(FPT) ) ); TEST_POINTER( d_vec );
	FPT * d_res; HANDLE_CUDA_ERROR( cudaMalloc( &d_res,         nrows * sizeof(FPT) ) ); TEST_POINTER( d_res );
	// copy necessary arrays to device
	HANDLE_CUDA_ERROR( cudaMemcpy( d_val, matCSR.val,           nnz * sizeof(FPT), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_col, matCSR.col,           nnz * sizeof(int), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_row, matCSR.row, ( nrows + 1 ) * sizeof(int), cudaMemcpyHostToDevice ) );
	HANDLE_CUDA_ERROR( cudaMemcpy( d_vec, vec,                nrows * sizeof(FPT), cudaMemcpyHostToDevice ) );
        // create handlers for cuSPARSE
        HANDLE_CUSPARSE_ERROR( cusparseCreate(&cusparseH) );
        HANDLE_CUSPARSE_ERROR( cusparseCreateCsr(&cusparseMD, matCSR.nrows, matCSR.nrows, matCSR.nnz,
                                      (void *)d_row, (void *)d_col, (void *)d_val,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, cudaDT ));
        HANDLE_CUSPARSE_ERROR( cusparseCreateDnVec(&cusparseVD1, matCSR.nrows, (void*)d_vec, cudaDT ));
        HANDLE_CUSPARSE_ERROR( cusparseCreateDnVec(&cusparseVD2, matCSR.nrows, d_res, cudaDT ));
	// get space buffer for cusparseSpMV
	HANDLE_CUSPARSE_ERROR( cusparseSpMV_bufferSize(cusparseH, cusparseO,
                    (void *)&one, cusparseMD, cusparseVD1, (void *)&zero,
                    cusparseVD2, cudaDT, cusparseAM, &cudaSpaceBufferSize ));
	void * cudaSpaceBuffer; HANDLE_CUDA_ERROR( cudaMalloc( &cudaSpaceBuffer, cudaSpaceBufferSize ) );
	// create events for time measuring
	cudaEvent_t cet1; HANDLE_CUDA_ERROR( cudaEventCreate( &cet1 ) );
	cudaEvent_t cet2; HANDLE_CUDA_ERROR( cudaEventCreate( &cet2 ) );
	// timed iterations
	float ti = 0.0, tt = 0.0;
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		HANDLE_CUDA_ERROR( cudaEventRecord( cet1 ) );
		HANDLE_CUSPARSE_ERROR( cusparseSpMV( cusparseH, cusparseO,
                      (void *)&one, cusparseMD, cusparseVD1, (void *)&zero,
                      cusparseVD2, cudaDT, cusparseAM, cudaSpaceBuffer));
		HANDLE_CUDA_ERROR( cudaEventRecord( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventSynchronize( cet2 ) );
		HANDLE_CUDA_ERROR( cudaEventElapsedTime( &ti, cet1, cet2 ) );
		tt = tt + ti;
	}
	// destroy events for time measuring
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet1 ) );
	HANDLE_CUDA_ERROR( cudaEventDestroy( cet2 ) );
	// copy result from device
	FPT * res = (FPT *) calloc( nrows, sizeof(FPT) ); TEST_POINTER( res );
	HANDLE_CUDA_ERROR( cudaMemcpy( res, d_res, nrows * sizeof(FPT), cudaMemcpyDeviceToHost ) );
	// free device memory
	HANDLE_CUDA_ERROR( cudaFree( d_val ) );
	HANDLE_CUDA_ERROR( cudaFree( d_col ) );
	HANDLE_CUDA_ERROR( cudaFree( d_row ) );
	HANDLE_CUDA_ERROR( cudaFree( d_vec ) );
	HANDLE_CUDA_ERROR( cudaFree( d_res ) );
	// store results
	str_res sr;
	strcpy( sr.name, "gcucsr" );
	sr.et    = ( (double) tt / (double) NUM_ITE ) * 1e-3;
	sr.ot    = 0.0;
	sr.flops = ( (double) matCSR.nnz * 2.0 ) / sr.et;
	get_errors( matCSR.nrows, ref, res, &(sr.sErr) );
	// free cpu memory
	free( res );
	return( sr );
}


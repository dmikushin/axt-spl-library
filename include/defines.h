#ifndef AXT_DEFINES_H
#define AXT_DEFINES_H

#define axt_stringify_value(a) axt_stringify(a)
#define axt_stringify(a) #a

#ifdef _OMP_
	#include <omp.h>
	#ifndef OMP_SCH
		#define OMP_SCH static
	#endif
#endif



#ifndef FP_FLOAT
	#define FP_FLOAT  1
#endif



#ifndef FP_DOUBLE
	#define FP_DOUBLE 2
#endif



#if FP_TYPE == FP_FLOAT
	#define FPT float
#endif



#if FP_TYPE == FP_DOUBLE
	#define FPT double
#endif



#ifndef UIN
	typedef unsigned int UIN;
#endif



#ifndef HDL
	#define HDL { fflush(stdout); printf( "---------------------------------------------------------------------------------------------------------\n" ); fflush(stdout); }
#endif



#ifndef BM
	#define BM { fflush(stdout); printf( "\nFile: %s    Line: %d.\n", __FILE__, __LINE__ ); fflush(stdout); }
#endif



#ifndef NUM_ITE
	#define NUM_ITE 250
#endif



#ifndef TILE_HW
	#define TILE_HW 32
#endif



#ifndef CHUNK_SIZE
	#define CHUNK_SIZE 32
#endif

typedef struct { double aErr; double rErr; UIN pos; } str_err;

typedef struct { char name[48]; double et; double ot; double flops; str_err sErr; } str_res;

#ifndef HANDLE_CUDA_ERROR
        #define HANDLE_CUDA_ERROR( ceID ) { if ( ceID != cudaSuccess ) { printf( "FILE: %s LINE: %d CUDA_ERROR: %s\n", __FILE__, __LINE__, cudaGetErrorString( ceID ) ); fflush(stdout); printf( "\nvim %s +%d\n", __FILE__, __LINE__); exit( EXIT_FAILURE ); } }
#endif



#ifndef TEST_POINTER
        #define TEST_POINTER( p ) { if ( p == NULL ) { fflush(stdout); printf( "\nFile: %s Line: %d Pointer: %s is null\n", __FILE__, __LINE__, #p ); fflush(stdout); exit( EXIT_FAILURE ); } }
#endif



typedef struct { char name[48]; UIN nrows; UIN nnz; UIN rmin; FPT rave; UIN rmax; FPT rsd; UIN bw; FPT * val; UIN * row; UIN * rowStart; UIN * rowEnd; UIN * col; UIN * rl; } str_matCSR;

#endif // AXT_DEFINES_H


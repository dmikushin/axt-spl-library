#ifndef AXT_DEFINES_H
#define AXT_DEFINES_H

#ifdef _OMP_
	#include <omp.h>
	#ifndef OMP_SCH
		#define OMP_SCH static
		char omp_schedule[7] = "static";
	#endif
#endif



#ifndef FP_FLOAT
	#define FP_FLOAT  1
#endif



#ifndef FP_DOUBLE
	#define FP_DOUBLE 2
#endif



#if FP_TYPE == FP_FLOAT
	typedef float  FPT;
	char fptMsg[6] = "float";
#endif



#if FP_TYPE == FP_DOUBLE
	typedef double FPT;
	char fptMsg[7] = "double";
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

#endif // AXT_DEFINES_H


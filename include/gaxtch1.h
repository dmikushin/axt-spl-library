#ifndef LIBAXT_GAXTCH1_H
#define LIBAXT_GAXTCH1_H

#include "defines.h"
#include "axt.h"

#define GAXTCH1_FULL_MASK 0xffffffff

__global__ void gaxtch1( const UIN LOG, const FPT * ax, const UIN * hdr, FPT * y )
{
	const UIN tidBLCK = threadIdx.x;
	const UIN widBLCK = tidBLCK >> 5;
	const UIN tidWARP = tidBLCK & 31;
	const UIN pAX     = blockIdx.x * 2 * blockDim.x + widBLCK * 64 + tidWARP;
	const UIN ro      = hdr[blockIdx.x * blockDim.x + tidBLCK];
	      UIN r, o, i;
	       __shared__ FPT blk1[32];
	extern __shared__ FPT blk2[];
	      FPT vo = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0;
	// initialize auxiliary arrays
	blk1[tidWARP] = 0.0;
	blk2[tidBLCK] = 0.0;
	__syncthreads();
	// read values from global memory array ax[] and perform multiplication on registers
	vo = ax[pAX] * ax[pAX+32];
	v1 = vo;
	__syncthreads();
	// perform warp-level reduction in v1
	for ( i = 1; i <=16; i = i * 2 )
	{
		v2 = __shfl_up_sync( GAXTCH1_FULL_MASK, v1, i );
		if (tidWARP >= i) v1 = v1 + v2;
	}
	__syncthreads();
	// store warp-level results on shared memory block blk1[]
	if (tidWARP == 31) blk1[widBLCK] = v1;
	__syncthreads();
	// use block's warp 0 to perform the reduction of the partial results stored on sb[]
	if (widBLCK == 0)
	{
		v2 = blk1[tidWARP];
		for ( i = 1; i <=16; i = i * 2 )
		{
			v3 = __shfl_up_sync( GAXTCH1_FULL_MASK, v2, i );
			if (tidWARP >= i) v2 = v2 + v3;
		}
		blk1[tidWARP] = v2;
	}
	__syncthreads();
	// update v1 with partial reductions from block's warp 0
	if (widBLCK > 0) v1 = v1 + blk1[widBLCK-1];
	__syncthreads();
	// write in blk2[] complete reduction values in v1
	blk2[tidBLCK] = v1;
	__syncthreads();
	// perform atomic addition to acumulate value in y[]
	if (ro)
	{
		r  = ro >> LOG;
		o  = ro & (blockDim.x - 1);
		v1 = blk2[tidBLCK + o] - v1 + vo;
		atomicAdd( &y[r], v1 );
	}
}

#endif // LIBAXT_GAXTCH1_H


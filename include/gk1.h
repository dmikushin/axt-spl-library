#ifndef LIBAXT_GK1_H
#define LIBAXT_GK1_H

#include "defines.h"
#include "axt.h"

__global__ void gk1( const int NROWS, const FPT * val, const UIN * col, const UIN * nmc, const UIN * chp, const UIN * permi, const FPT * x, FPT * y )
{
	const UIN gid = blockIdx.x * blockDim.x + threadIdx.x;
	const UIN lid = threadIdx.x;
	const UIN cid = gid / CHUNK_SIZE;
	const UIN wid = lid & ( CHUNK_SIZE - 1 );
	if ( gid < NROWS )
	{
		UIN to = chp[cid] + wid;
		UIN ul = nmc[cid] * CHUNK_SIZE + to;
		FPT sum = val[to] * x[col[to]];
		for ( to = ( to + CHUNK_SIZE ); to < ul; to = ( to + CHUNK_SIZE ) )
			sum = sum + val[to] * x[col[to]];
		y[permi[gid]] = sum;
	}
}

#endif // LIBAXT_GK1_H

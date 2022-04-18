#ifndef LIBAXT_GAXC_H
#define LIBAXT_GAXC_H

#include "defines.h"
#include "axt.h"

#define GAXC_FULL_MASK 0xffffffff

static __global__ void gaxc( const UIN NROWS, const FPT * ax, const UIN * brp, FPT * y )
{
	const UIN tidGRID = blockIdx.x * blockDim.x + threadIdx.x;
	const UIN widGRID = tidGRID >> 5;
	if ( widGRID < NROWS )
	{
		const UIN tidWARP = tidGRID & 31;
		const UIN p1      = brp[widGRID]   + tidWARP;
		const UIN p2      = brp[widGRID+1] + tidWARP;
		      UIN pAX;
		      FPT val = 0.0, red = 0.0;
		for ( pAX = p1; pAX < p2; pAX = pAX + 64 )
		{
			val = ax[pAX] * ax[pAX+32];
			val = val + __shfl_down_sync( GAXC_FULL_MASK, val, 16 );
			val = val + __shfl_down_sync( GAXC_FULL_MASK, val,  8 );
			val = val + __shfl_down_sync( GAXC_FULL_MASK, val,  4 );
			val = val + __shfl_down_sync( GAXC_FULL_MASK, val,  2 );
			val = val + __shfl_down_sync( GAXC_FULL_MASK, val,  1 );
			red = red + val;
		}
		if (tidWARP == 0) y[widGRID] = red;
	}
}

#endif // LIBAXT_GAXC_H


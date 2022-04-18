#ifndef LIBAXT_GAXTUH1HW16W
#define LIBAXT_GAXTUH1HW16W

#include "defines.h"
#include "axt.h"

#define GAXTUH1HW16W_FULL_MASK 0xffffffff

__global__ void gaxtuh1hw16w( const UIN TPW, const UIN TN, const FPT * ax, const UIN * rwp, FPT * y )
{
	const UIN tidGRID = blockIdx.x * blockDim.x + threadIdx.x;
	const UIN tidWARP = tidGRID & 31;
	const UIN ll      = (tidGRID >> 5) * TPW;
	const UIN ul      = ll + TPW;
	      UIN t1, t2, r1, r2, p_ax1, p_ax2;
	      FPT v0, v1, v2;
	for ( t1 = ll; t1 < ul; t1 = t1 + 2 )
	{
		t2    = t1 + 1;
		r1    = rwp[t1];
		r2    = rwp[t2];
		p_ax1 = t1 * 32 + tidWARP;
		p_ax2 = t2 * 32 + tidWARP;
		v1    = ax[p_ax1];
		v1    = v1 * __shfl_down_sync( GAXTUH1HW16W_FULL_MASK, v1, 16 );
		v2    = ax[p_ax2];
		v2    = v2 * __shfl_up_sync  ( GAXTUH1HW16W_FULL_MASK, v2, 16 );
		if (tidWARP < 16) v0 = v1;
		else              v0 = v2;
		v0    = v0 + __shfl_down_sync( GAXTUH1HW16W_FULL_MASK, v0, 8  );
		v0    = v0 + __shfl_down_sync( GAXTUH1HW16W_FULL_MASK, v0, 4  );
		v0    = v0 + __shfl_down_sync( GAXTUH1HW16W_FULL_MASK, v0, 2  );
		v0    = v0 + __shfl_down_sync( GAXTUH1HW16W_FULL_MASK, v0, 1  );
		     if (tidWARP ==  0) atomicAdd( &y[r1], v0 );
		else if (tidWARP == 16) atomicAdd( &y[r2], v0 );
	}
}

#endif // LIBAXT_GAXTUH1HW16W


#ifndef LIBAXT_GAXTUH1HW08W_H
#define LIBAXT_GAXTUH1HW08W_H

#include "defines.h"
#include "axt.h"

#define GAXTUH1HW08W_FULL_MASK 0xffffffff

__global__ void gaxtuh1hw08w( const UIN TPW, const UIN TN, const FPT * ax, const UIN * rwp, FPT * y )
{
	const UIN tidGRID = blockIdx.x * blockDim.x + threadIdx.x;
	const UIN tidWARP = tidGRID & 31;
	const UIN tid16   = tidGRID & 15;
	const UIN ll      = (tidGRID >> 5) * TPW;
	const UIN ul      = ll + TPW;
	      UIN t1, t2, t3, t4, r1, r2, r3, r4, p_ax1, p_ax2, p_ax3, p_ax4;
	      FPT v0, v1, v2, v3, v4;
	for ( t1 = ll; t1 < ul; t1 = t1 + 4 )
	{
		t2    = t1 + 1;
		t3    = t1 + 2;
		t4    = t1 + 3;
		r1    = rwp[t1];
		r2    = rwp[t2];
		r3    = rwp[t3];
		r4    = rwp[t4];
		p_ax1 = t1 * 16 + tid16;
		p_ax2 = t2 * 16 + tid16;
		p_ax3 = t3 * 16 + tid16;
		p_ax4 = t4 * 16 + tid16;
		v1    = ax[p_ax1];
		v1    = v1 * __shfl_down_sync( GAXTUH1HW08W_FULL_MASK, v1, 8 );
		v2    = ax[p_ax2];
		v2    = v2 * __shfl_up_sync  ( GAXTUH1HW08W_FULL_MASK, v2, 8 );
		v3    = ax[p_ax3];
		v3    = v3 * __shfl_up_sync  ( GAXTUH1HW08W_FULL_MASK, v3, 8 );
		v4    = ax[p_ax4];
		v4    = v4 * __shfl_up_sync  ( GAXTUH1HW08W_FULL_MASK, v4, 8 );
		     if                      (tidWARP <  8)   v0 = v1;
		else if ( (tidWARP >=  8) && (tidWARP < 16) ) v0 = v2;
		else if ( (tidWARP >= 16) && (tidWARP < 24) ) v0 = v3;
		else                                          v0 = v4;
		v0    = v0 + __shfl_down_sync( GAXTUH1HW08W_FULL_MASK, v0, 4  );
		v0    = v0 + __shfl_down_sync( GAXTUH1HW08W_FULL_MASK, v0, 2  );
		v0    = v0 + __shfl_down_sync( GAXTUH1HW08W_FULL_MASK, v0, 1  );
		     if (tidWARP ==  0) atomicAdd( &y[r1], v0 );
		else if (tidWARP ==  8) atomicAdd( &y[r2], v0 );
		else if (tidWARP == 16) atomicAdd( &y[r3], v0 );
		else if (tidWARP == 24) atomicAdd( &y[r4], v0 );
	}
}

#endif // LIBAXT_GAXTUH1HW08W_H


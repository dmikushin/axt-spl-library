#ifndef LIBAXT_GAXTUH1HW04W_H
#define LIBAXT_GAXTUH1HW04W_H

#include "defines.h"
#include "axt.h"

#define GAXTUH1HW04W_FULL_MASK 0xffffffff

__global__ void gaxtuh1hw04w( const UIN TPW, const UIN TN, const FPT * ax, const UIN * rwp, FPT * y )
{
	const UIN tidGRID = blockIdx.x * blockDim.x + threadIdx.x;
	const UIN tidWARP = tidGRID & 31;
	const UIN tid8    = tidGRID &  7;
	const UIN ll      = (tidGRID >> 5) * TPW;
	const UIN ul      = ll + TPW;
	      UIN t1, t2, t3, t4, t5, t6, t7, t8;
	      UIN r1, r2, r3, r4, r5, r6, r7, r8;
	      UIN p_ax1, p_ax2, p_ax3, p_ax4, p_ax5, p_ax6, p_ax7, p_ax8;
	      FPT v0, v1, v2, v3, v4, v5, v6, v7, v8;
	for ( t1 = ll; t1 < ul; t1 = t1 + 8 )
	{
		t2    = t1 + 1;
		t3    = t1 + 2;
		t4    = t1 + 3;
		t5    = t1 + 4;
		t6    = t1 + 5;
		t7    = t1 + 6;
		t8    = t1 + 7;
		r1    = rwp[t1];
		r2    = rwp[t2];
		r3    = rwp[t3];
		r4    = rwp[t4];
		r5    = rwp[t5];
		r6    = rwp[t6];
		r7    = rwp[t7];
		r8    = rwp[t8];
		p_ax1 = t1 * 8 + tid8;
		p_ax2 = t2 * 8 + tid8;
		p_ax3 = t3 * 8 + tid8;
		p_ax4 = t4 * 8 + tid8;
		p_ax5 = t5 * 8 + tid8;
		p_ax6 = t6 * 8 + tid8;
		p_ax7 = t7 * 8 + tid8;
		p_ax8 = t8 * 8 + tid8;
		v1    = ax[p_ax1];
		v1    = v1 * __shfl_down_sync( GAXTUH1HW04W_FULL_MASK, v1, 4 );
		v2    = ax[p_ax2];
		v2    = v2 * __shfl_up_sync  ( GAXTUH1HW04W_FULL_MASK, v2, 4 );
		v3    = ax[p_ax3];
		v3    = v3 * __shfl_up_sync  ( GAXTUH1HW04W_FULL_MASK, v3, 4 );
		v4    = ax[p_ax4];
		v4    = v4 * __shfl_up_sync  ( GAXTUH1HW04W_FULL_MASK, v4, 4 );
		v5    = ax[p_ax5];
		v5    = v5 * __shfl_up_sync  ( GAXTUH1HW04W_FULL_MASK, v5, 4 );
		v6    = ax[p_ax6];
		v6    = v6 * __shfl_up_sync  ( GAXTUH1HW04W_FULL_MASK, v6, 4 );
		v7    = ax[p_ax7];
		v7    = v7 * __shfl_up_sync  ( GAXTUH1HW04W_FULL_MASK, v7, 4 );
		v8    = ax[p_ax8];
		v8    = v8 * __shfl_up_sync  ( GAXTUH1HW04W_FULL_MASK, v8, 4 );
		     if                      (tidWARP <  4)   v0 = v1;
		else if ( (tidWARP >=  4) && (tidWARP <  8) ) v0 = v2;
		else if ( (tidWARP >=  8) && (tidWARP < 12) ) v0 = v3;
		else if ( (tidWARP >= 12) && (tidWARP < 16) ) v0 = v4;
		else if ( (tidWARP >= 16) && (tidWARP < 20) ) v0 = v5;
		else if ( (tidWARP >= 20) && (tidWARP < 24) ) v0 = v6;
		else if ( (tidWARP >= 24) && (tidWARP < 28) ) v0 = v7;
		else                                          v0 = v8;
		    v0    = v0 + __shfl_down_sync( GAXTUH1HW04W_FULL_MASK, v0, 2  );
		    v0    = v0 + __shfl_down_sync( GAXTUH1HW04W_FULL_MASK, v0, 1  );
		     if (tidWARP ==  0) atomicAdd( &y[r1], v0 );
		else if (tidWARP ==  4) atomicAdd( &y[r2], v0 );
		else if (tidWARP ==  8) atomicAdd( &y[r3], v0 );
		else if (tidWARP == 12) atomicAdd( &y[r4], v0 );
		else if (tidWARP == 16) atomicAdd( &y[r5], v0 );
		else if (tidWARP == 20) atomicAdd( &y[r6], v0 );
		else if (tidWARP == 24) atomicAdd( &y[r7], v0 );
		else if (tidWARP == 28) atomicAdd( &y[r8], v0 );
	}
}

#endif // LIBAXT_GAXTUH1HW04W_H


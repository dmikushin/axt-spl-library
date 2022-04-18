#ifndef LIBAXT_GATXCH_H
#define LIBAXT_GATXCH_H

#include "defines.h"
#include "axt.h"

__global__ void gaxtch( const UIN LOG, const UIN TH, const FPT * ax, const UIN * hdr, FPT * y )
{
	const UIN tidGRID = blockIdx.x * blockDim.x + threadIdx.x;
	const UIN widGRID = tidGRID >> 5;
	const UIN tidWARP = tidGRID & 31;
	const UIN THS     = TH * 32;
	const UIN ul_hdr  = (widGRID + 1) * THS;
	const UIN TS      = TH * 64;
	const UIN ul_ax   = (widGRID + 1) * TS;
	      UIN a1_hdr, ro, r, o, a1_ax, a2_ax, p_ax;
	      FPT red;
	a1_hdr = widGRID * THS + tidWARP;
	ro     = hdr[a1_hdr];
	r      = ro >> LOG;
	o      = (ro & (TH-1)) * 64;
	a1_ax  = widGRID * TS + tidWARP;
	a2_ax  = a1_ax + o;
	do {
		red    = 0.0;
		for ( p_ax = a1_ax; p_ax <= a2_ax; p_ax = p_ax + 64 )
		{
			red    = red + ax[p_ax] * ax[p_ax+32];
			a1_hdr = a1_hdr + 32;
		}
		atomicAdd( &y[r], red );
		if (a1_hdr < ul_hdr)
		{
			ro = hdr[a1_hdr];
			r      = ro >> LOG;
			o      = (ro & (TH-1)) * 64;
			a1_ax  = p_ax;
			a2_ax  = a1_ax + o;
		}
	} while (p_ax < ul_ax);
}

#endif // LIBAXT_GATXCH_H


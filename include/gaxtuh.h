#ifndef LIBAXT_GAXTUH_H
#define LIBAXT_GAXTUH_H

#include "defines.h"
#include "axt.h"

#if 0
__global__ void gaxtuh( const UIN TN, const UIN TH, const FPT * ax, const UIN * rwp, FPT * y )
{
	const UIN tidGRID = blockIdx.x * blockDim.x + threadIdx.x;
	const UIN widGRID = tidGRID >> 5;
	if ( widGRID < TN )
	{
		const UIN tidWARP = tidGRID & 31;
		const UIN rid     = rwp[widGRID*32 + tidWARP];
		const UIN p1      = widGRID * TH * 64 + tidWARP;
		const UIN p2      = p1 + TH * 64;
		      UIN pAX     = p1;
		      FPT val     = ax[pAX] * ax[pAX+32];
		for ( pAX = pAX + 64; pAX < p2; pAX = pAX + 64 )
			val = val + ax[pAX] * ax[pAX+32];
		atomicAdd( &y[rid], val );
	}
	return;
}
#else
__global__ void gaxtuh( const UIN TPW, const UIN TN, const UIN TH, const FPT * ax, const UIN * rwp, FPT * y )
{
	const UIN tidGRID = blockIdx.x * blockDim.x + threadIdx.x;
	const UIN tidWARP = tidGRID & 31;
	const UIN widGRID = (tidGRID >> 5);
	const UIN ll      = widGRID * TPW;
	const UIN ul      = ll + TPW;
	      UIN t, p1, p2, p_ax, r;
	      FPT val;
	for ( t = ll; t < ul; t++ )
	{
		p1  = t * TH * 64 + tidWARP;
		p2  = p1 + TH * 64;
		val = 0.0;
		for ( p_ax = p1; p_ax < p2; p_ax = p_ax + 64 )
			val = val + ax[p_ax] * ax[p_ax+32];
		r = rwp[t*32 + tidWARP];
		atomicAdd( &y[r], val );
	}
}
#endif

#endif // LIBAXT_GAXTUH_H


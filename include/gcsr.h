#ifndef LIBAXT_GCSR_H
#define LIBAXT_GCSR_H

#include "defines.h"
#include "axt.h"

__global__ void gcsr( const UIN nrows, const FPT * val, const UIN * col, const UIN * row, const FPT * x, FPT * y )
{
	const UIN rowID = blockIdx.x * blockDim.x + threadIdx.x;
	if ( rowID < nrows )
	{
		UIN i;
		FPT aux = 0.0;
		for ( i = row[rowID]; i < row[rowID + 1]; i++ )
			aux = aux + val[i] * x[col[i]];
		y[rowID] = aux;
	}
}

#endif // LIBAXT_GCSR_H


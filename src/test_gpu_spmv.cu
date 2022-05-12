// program: cudaSpmv.cu
// author: Edoardo Coronado
// date: 21-08-2019 (dd-mm-yyyy)

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <map>
#include <math.h>
#include <vector>

#include "defines.h"
#include "diff.h"
#include "axt.h"
#include "axc.h"
#include "k1.h"
#include "ncsr.h"

typedef struct { UIN cbs; char matFileName[48]; UIN ompMT; } str_inputArgs;

static str_inputArgs checkArgs( const UIN argc, char ** argv )
{
	if ( argc < 3 )
	{
		fflush(stdout);
		printf( "\n\tMissing input arguments.\n" );
		printf( "\n\tUsage:\n\n\t\t%s <cudaBlockSize> <matFileName>\n\n", argv[0] );
		printf( "\t\t\t<cudaBlockSize>:  number of threads per cuda block.\n" );
		printf( "\t\t\t<matFileName>:    file's name that contains the matrix in CSR format [string].\n" );
		fflush(stdout);
		exit( EXIT_FAILURE );
	}
	str_inputArgs sia;
	sia.cbs    = atoi( argv[1] );
	strcpy( sia.matFileName, argv[2] );
	sia.ompMT = 1;
	#pragma omp parallel if(_OPENMP)
	{
		#pragma omp master
		{
			sia.ompMT = omp_get_max_threads();
		}
	}
	return( sia );
}



#ifndef ABORT
	#define ABORT { fflush(stdout); printf( "\nFile: %s Line: %d execution is aborted.\n", __FILE__, __LINE__ ); fflush(stdout); exit( EXIT_FAILURE ); }
#endif



static void printRunSettings( const str_inputArgs sia )
{
	time_t t = time(NULL);
	struct tm tm = *localtime(&t);
	HDL; printf( "run settings\n" ); HDL;
	#ifdef _CIRRUS_
	printf( "hostname:           %s\n", "cirrus.EPCC" );
	#endif
	#ifdef _KAY_
	printf( "hostname:           %s\n", "kay.ICHEC" );
	#endif
	#ifdef _CTGPGPU2_
	printf( "hostname:           %s\n", "ctgpgpu2.CITIUS" );
	#endif
	printf( "srcFileName:        %s\n", __FILE__ );
	printf( "date:               %04d-%02d-%02d (yyyy-mm-dd)\n", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday );
	printf( "time:               %02d:%02d:%02d (hh:mm:ss)\n", tm.tm_hour, tm.tm_min, tm.tm_sec );
	printf( "matFileName:        %s\n", sia.matFileName );
	#ifdef _OMP_
	printf( "ompMaxThreads:      %d\n", sia.ompMT );
	printf( "omp_schedule:       " axt_stringify_value(OMP_SCH) "\n" );
	#endif
	printf( "FPT:                " axt_stringify_value(FPT) "\n" );
	printf( "sizeof(FPT):        %zu bytes\n", sizeof(FPT) );
	printf( "cudaBlockSize:      %d\n",  sia.cbs  );
	printf( "NUM_ITE:            %d\n", (UIN) NUM_ITE );
	printf( "CHUNK_SIZE:         %d\n", (UIN) CHUNK_SIZE );
	printf( "TILE_HW:            %d\n", (UIN) TILE_HW ); fflush(stdout);
	return;
}


static str_matCSR matrixReading( const char * matFileName )
{
	str_matCSR matCSR;
	strcpy( matCSR.name, matFileName );
	if ( strstr( matFileName, ".csr" ) != NULL )
	{
		FILE * fh;
		fh = fopen( matFileName, "r" );
		if ( fh == NULL )
		{
			printf( "\nmatrixReading is unable to open .csr file\n\n" );
			exit( EXIT_FAILURE );
		}
		if ( fscanf( fh, "%d %d", &(matCSR.nrows), &(matCSR.nnz) ) != 2 ) ABORT;
		matCSR.val = (FPT *) malloc(   matCSR.nnz        * sizeof(FPT) ); TEST_POINTER( matCSR.val );
		matCSR.col = (UIN *) malloc(   matCSR.nnz        * sizeof(UIN) ); TEST_POINTER( matCSR.col );
		matCSR.row = (UIN *) malloc( ( matCSR.nrows + 1) * sizeof(UIN) ); TEST_POINTER( matCSR.row );
		matCSR.rl  = (UIN *) malloc(   matCSR.nrows      * sizeof(UIN) ); TEST_POINTER( matCSR.rl  );
		int i;
		for ( i = 0; i < ( matCSR.nnz ); i++ )
		{
			#if FP_TYPE == FPT_FLOAT
				if ( fscanf( fh, "%f %d\n",  &( matCSR.val[i] ), &( matCSR.col[i] ) ) != 2 ) ABORT;
			#else
				if ( fscanf( fh, "%lf %d\n", &( matCSR.val[i] ), &( matCSR.col[i] ) ) != 2 ) ABORT;
			#endif
		}
		for ( i = 0; i < ( matCSR.nrows + 1 ); i++ )
			if ( fscanf( fh, "%d", &(matCSR.row[i]) ) != 1 ) ABORT;
		fclose( fh );
	}
	else if ( strstr( matFileName, ".bin" ) != NULL )
	{
		size_t aux = 0;
		FILE * fh;
		fh = fopen( matFileName, "r" );
		if ( fh == NULL )
		{
			printf( "\nmatrixReading is unable to open .bin file\n\n" );
			exit( EXIT_FAILURE );
		}
		aux = fread( &(matCSR.nrows), sizeof(UIN), 1, fh );
		aux = fread( &(matCSR.nnz),   sizeof(UIN), 1, fh );
		matCSR.val = (FPT *) malloc(   matCSR.nnz        * sizeof(FPT) ); TEST_POINTER( matCSR.val );
		matCSR.col = (UIN *) malloc(   matCSR.nnz        * sizeof(UIN) ); TEST_POINTER( matCSR.col );
		matCSR.row = (UIN *) malloc( ( matCSR.nrows + 1) * sizeof(UIN) ); TEST_POINTER( matCSR.row );
		matCSR.rl  = (UIN *) malloc(   matCSR.nrows      * sizeof(UIN) ); TEST_POINTER( matCSR.rl  );
		aux = fread( matCSR.val, sizeof(FPT),   matCSR.nnz,         fh );
		aux = fread( matCSR.col, sizeof(UIN),   matCSR.nnz,         fh );
		aux = fread( matCSR.row, sizeof(UIN), ( matCSR.nrows + 1 ), fh );
		aux++;
		fclose(fh);
	}
	else
	{
		char buffer[128];
		strcpy( buffer, "matrixReading detected that " );
		strcat( buffer, matFileName );
		strcat( buffer, " has NOT .csr or .bin extension" );
		printf( "\n%s\n\n", buffer );
		exit( EXIT_FAILURE );
	}
	return( matCSR );
}



static void printMatrixStats( const char * matFileName, str_matCSR * matCSR )
{
	UIN    i, rl, rmin = 1e9, rmax = 0, j, bw = 0;
	int    dif;
	double rave1 = 0.0, rave2 = 0.0, rsd = 0.0;
	for ( i = 0; i < matCSR->nrows; i++ )
	{
		rl            = matCSR->row[i + 1] - matCSR->row[i];
		matCSR->rl[i] = rl;
		rave1         = rave1 +   rl;
		rave2         = rave2 + ( rl * rl );
		rmin          = (rmin<rl) ? rmin : rl;
		rmax          = (rmax>rl) ? rmax : rl;
		for ( j = matCSR->row[i]; j < matCSR->row[i+1]; j++ )
		{
			dif = abs( ((int) i) - ((int) matCSR->col[j]) );
			bw  = ( dif > bw ) ? dif : bw ;
		}
	}
	rave1 = rave1 / (double) (matCSR->nrows);
	rave2 = rave2 / (double) (matCSR->nrows);
	rsd   = sqrt( rave2 - ( rave1 * rave1 ) );
	matCSR->rmin = rmin;
	matCSR->rave = rave1;
	matCSR->rmax = rmax;
	matCSR->rsd  = rsd;
	matCSR->bw   = bw;
	char name[64];
	strcpy( name, matFileName );
	char * token1;
	const char deli[2] = ".";
	token1 = strtok( name, deli );
	strcat( token1, ".sta" );
	FILE * fh;
	fh = fopen( name, "w+" );
	fprintf( fh, "------------------------------------\n");
	fprintf( fh, "matrix's statistics\n");
	fprintf( fh, "------------------------------------\n");
	fprintf( fh, "name:  %28s\n",    matFileName );
	fprintf( fh, "nrows: %28d\n",    matCSR->nrows );
	fprintf( fh, "nnz:   %28d\n",    matCSR->nnz );
	fprintf( fh, "rmin:  %28d\n",    matCSR->rmin );
	fprintf( fh, "rave:  %28.2lf\n", matCSR->rave );
	fprintf( fh, "rmax:  %28d\n",    matCSR->rmax );
	fprintf( fh, "rsd:   %28.2lf\n", matCSR->rsd );
	fprintf( fh, "rsdp:  %28.2lf\n", ( ( rsd / rave1 ) * 100 ) );
	fprintf( fh, "bw:    %28d\n",    matCSR->bw );
	fclose( fh );
	return;
}



typedef struct { char name[48]; double mfp; double beta; double ct; } str_formatData;



#ifndef GT
	#define GT( t ) { clock_gettime( CLOCK_MONOTONIC, &t ); }
#endif



static double measure_time( const struct timespec t2, const struct timespec t1 )
{
	double t = (double) ( t2.tv_sec - t1.tv_sec ) + ( (double) ( t2.tv_nsec - t1.tv_nsec ) ) * 1e-9;
	return( t );
}



static str_formatData getFormatDataCSR( str_matCSR * matCSR )
{
	// define local variables
	UIN i, ii;
	double ti = 0.0, tt = 0.0;
	struct timespec t1, t2;
	matCSR->rowStart = (UIN *) calloc( matCSR->nrows, sizeof(UIN) ); TEST_POINTER( matCSR->rowStart );
	matCSR->rowEnd   = (UIN *) calloc( matCSR->nrows, sizeof(UIN) ); TEST_POINTER( matCSR->rowEnd   );
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		for ( ii = 0; ii < matCSR->nrows; ii++ )
		{
			matCSR->rowStart[ii] = matCSR->row[ii];
			matCSR->rowEnd[ii]   = matCSR->row[ii+1];
		}
		GT( t2 );
		ti = measure_time( t2, t1 );
		tt = tt + ti;
	}
	// format's name
	str_formatData fd;
	strcpy( fd.name, "fcsr" );
	// CSR memory footprint
	fd.mfp =          (double) (   matCSR->nnz         * sizeof(FPT) ); // val
	fd.mfp = fd.mfp + (double) (   matCSR->nnz         * sizeof(UIN) ); // col
	fd.mfp = fd.mfp + (double) ( ( matCSR->nrows + 1 ) * sizeof(UIN) ); // row
	fd.mfp = fd.mfp + (double) (   matCSR->nrows       * sizeof(FPT) ); // vec
	// CSR occupancy ( beta )
	fd.beta = ( (double) matCSR->nnz / (double) matCSR->nnz );
	// CSR conversion time (conversion time for MKL functions)
	fd.ct = tt / (double) NUM_ITE;
	return( fd );
}

static void init_vec( const UIN ompNT, const UIN len, FPT * vec )
{
	UIN i;
	#pragma omp parallel for default(shared) private(i) num_threads(ompNT) schedule(OMP_SCH) if(_OPENMP)
	for ( i = 0 ; i < len; i++ )
		vec[i] = (FPT) i;
	return;
}

str_res test_ncsr( const UIN ompNT, const str_matCSR matCSR, const FPT * vec, const FPT * ref );

str_res test_gcsr( const UIN cudaBlockSize, const str_matCSR matCSR, const FPT * vec, const FPT * ref );

typedef struct { UIN ind; UIN val; } str_pair;

static int orderFunction( const void * ele1, const void * ele2 )
{
       return (  ( (str_pair *) ele2 )->val - ( (str_pair *) ele1 )->val  );
}

static void getArrayPermiK1( const str_matCSR matCSR, str_matK1 * matK1 )
{
       str_pair * list = (str_pair *) malloc( matCSR.nrows * sizeof(str_pair) ); TEST_POINTER( list );
       UIN i;
       for ( i = 0; i < matK1->nrows; i++ )
       {
               list[i].ind = i;
               list[i].val = matCSR.rl[i];
       }
       qsort( list, matK1->nrows, sizeof(str_pair), orderFunction );
       for ( i = 0; i < matK1->nrows; i++ )
               matK1->permi[i] = list[i].ind;
       free( list );
       return;
}

static UIN getArraysNmcChpK1( const str_matCSR matCSR, str_matK1 * matK1 )
{
	UIN i, p, n, l = 0, chunkNum = ( matCSR.nrows + CHUNK_SIZE - 1 ) / CHUNK_SIZE;
	for ( i = 0 ; i < chunkNum; i++ )
	{
		p             = matK1->permi[i * CHUNK_SIZE];
		n             = matCSR.rl[p];
		matK1->nmc[i] = n;
		l             = l + CHUNK_SIZE * n;
	}
	for ( i = 1; i < matK1->chunkNum; i++ )
		matK1->chp[i] = matK1->chp[i-1] + ( matK1->nmc[i-1] * CHUNK_SIZE );
	return l;
}
 
static void getArraysValColK1( const str_matCSR matCSR, str_matK1 * matK1 )
{
	const UIN chunkNum = matK1->chunkNum;
	UIN chunkID, rid, row, posCSR, rowOff, posK1;
	for ( chunkID = 0; chunkID < chunkNum; chunkID++ )
	{
		for ( rid = 0; rid < CHUNK_SIZE; rid++ )
		{
			row = chunkID * CHUNK_SIZE + rid;
			if ( row == matCSR.nrows ) return;
			row = matK1->permi[row];
			for ( posCSR = matCSR.row[row], rowOff = 0; posCSR < matCSR.row[row + 1]; posCSR++, rowOff++ )
			{
				posK1             = matK1->chp[chunkID] + rowOff * CHUNK_SIZE + rid;
				matK1->val[posK1] = matCSR.val[posCSR];
				matK1->col[posK1] = matCSR.col[posCSR];
			}
		}
	}
	return;
}
 
static str_formatData getFormatDataK1( const UIN blockSize, const str_matCSR matCSR, const FPT * vec, str_matK1 * matK1 )
{
	// get K1 parameters
	matK1->nrows     = matCSR.nrows;
	matK1->nnz       = matCSR.nnz;
	matK1->chunkNum  = ( matCSR.nrows + CHUNK_SIZE - 1 ) / CHUNK_SIZE;
	matK1->permi     = (UIN *) calloc( ( matK1->chunkNum + 1 ) * CHUNK_SIZE, sizeof(UIN) ); TEST_POINTER( matK1->permi );
	matK1->nmc       = (UIN *) calloc(   matK1->chunkNum,                    sizeof(UIN) ); TEST_POINTER( matK1->nmc   );
	matK1->chp       = (UIN *) calloc(   matK1->chunkNum,                    sizeof(UIN) ); TEST_POINTER( matK1->chp   );
	UIN i;
	for ( i = 0; i < ( matK1->chunkNum + 1 ) * CHUNK_SIZE; i++ )
		matK1->permi[i] = 0;
	// get matK1
	struct timespec t1, t2;
	double ti = 0.0, tt = 0.0, tc = 0.0;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		getArrayPermiK1( matCSR, matK1 );
		GT( t2 );
		ti = measure_time( t2, t1 );
		tt = tt + ti;
	}
	ti = tt / (double) NUM_ITE;
	tc = tc + ti;
	tt = 0.0;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		matK1->lenVC = getArraysNmcChpK1( matCSR, matK1 );
		GT( t2 );
		ti = measure_time( t2, t1 );
		tt = tt + ti;
	}
	ti = tt / (double) NUM_ITE;
	tc = tc + ti;
	matK1->val = (FPT *) calloc( matK1->lenVC, sizeof(FPT) ); TEST_POINTER( matK1->val );
	matK1->col = (UIN *) calloc( matK1->lenVC, sizeof(UIN) ); TEST_POINTER( matK1->col );
	tt = 0.0;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		getArraysValColK1( matCSR, matK1 );
		GT( t2 );
		ti = measure_time( t2, t1 );
		tt = tt + ti;
	}
	ti = tt / (double) NUM_ITE;
	tc = tc + ti;
	// format's name
	str_formatData fd;
	strcpy( fd.name, "fk1" );
	// K1 memory footprint
	fd.mfp =          (double) ( matK1->chunkNum * sizeof(UIN) ); // nmc
	fd.mfp = fd.mfp + (double) ( matK1->chunkNum * sizeof(UIN) ); // chp
	fd.mfp = fd.mfp + (double) ( matK1->lenVC    * sizeof(FPT) ); // val
	fd.mfp = fd.mfp + (double) ( matK1->lenVC    * sizeof(UIN) ); // col
	fd.mfp = fd.mfp + (double) ( matK1->nrows    * sizeof(UIN) ); // permi
	fd.mfp = fd.mfp + (double) ( matK1->nrows    * sizeof(FPT) ); // vec
	// K1 occupancy ( beta )
	fd.beta = ( (double) matK1->nnz / (double) (matK1->lenVC) );
	// K1 conversion time
	fd.ct = tc;
	return( fd );
}

str_res test_gcucsr( const str_matCSR matCSR, const FPT * vec, const FPT * ref );

str_res test_gk1( const UIN cudaBlockSize, const str_matK1 matK1, const FPT * vec, const FPT * ref );

static UIN get_brpAXC( const str_matCSR matCSR, str_matAXC * matAXC )
{
	const UIN hbs   = matAXC->hbs;
	const UIN nrows = matAXC->nrows;
	      UIN rowID, brickNum;
	for ( rowID = 0; rowID < nrows; rowID++ )
	{
		brickNum               = ( matCSR.rl[rowID] + hbs - 1 ) / hbs;
		matAXC->brp[rowID + 1] = matAXC->brp[rowID]  + ( 2 * brickNum * hbs );
	}
	return( matAXC->brp[matAXC->nrows] );
}



static void get_axAXC( const UIN ompNT, const str_matCSR matCSR, const FPT * vec, str_matAXC * matAXC )
{
	const UIN hbs   = matAXC->hbs;
	const UIN nrows = matAXC->nrows;
	      UIN rowID, posAX, counter, posCSR;
	#pragma omp parallel for default(shared) private(rowID,posAX,counter,posCSR) num_threads(ompNT) schedule(OMP_SCH) if(_OPENMP)
	for ( rowID = 0; rowID < nrows; rowID++ )
	{
		posAX   = matAXC->brp[rowID];
		counter = 0;
		for ( posCSR = matCSR.row[rowID]; posCSR < matCSR.row[rowID + 1]; posCSR++ )
		{
			matAXC->ax[posAX]       = matCSR.val[posCSR];
			matAXC->ax[posAX + hbs] = vec[matCSR.col[posCSR]];
			if ( counter == (hbs - 1) )
			{
				posAX  = posAX + 1 + hbs;
				counter = 0;
			}
			else
			{
				posAX++;
				counter++;
			}
		}
	}
	return;
}



static void get_mapxAXC( const UIN ompNT, const str_matCSR matCSR, str_matAXC * matAXC )
{
	const UIN nrows = matAXC->nrows;
	      UIN rowID, pos1, pos2, pos, eleID;
	#pragma omp parallel for default(shared) private(rowID,pos1,pos2,pos,eleID) num_threads(ompNT) schedule(OMP_SCH) if(_OPENMP)
	for ( rowID = 0; rowID < nrows; rowID++ )
	{
		pos1 = matCSR.row[rowID];
		pos2 = matCSR.row[rowID+1];
		pos  = matAXC->brp[rowID]>>1;
		for ( eleID = pos1; eleID < pos2; eleID++ )
		{
			matAXC->mapx[pos] = matCSR.col[eleID];
			pos++;
		}
	}
	return;
}



static str_formatData getFormatDataAXC( const UIN ompNT, const UIN hbs, const str_matCSR matCSR, const FPT * vec, str_matAXC * matAXC )
{
	// get AXC parameters
	matAXC->nrows  = matCSR.nrows;
	matAXC->nnz    = matCSR.nnz;
	matAXC->hbs    = hbs;
	matAXC->lenBRP = matCSR.nrows + 1;
	matAXC->brp    = (UIN *) calloc( matAXC->lenBRP, sizeof(UIN) ); TEST_POINTER( matAXC->brp  );
	// get matAXC
	struct timespec t1, t2;
	double ti = 0.0, tt = 0.0, tc = 0.0;
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		matAXC->lenAX = get_brpAXC( matCSR, matAXC );
		GT( t2 );
		ti = measure_time( t2, t1 );
		tt = tt + ti;
	}
	ti = tt / (double) NUM_ITE;
	tc = tc + ti;
	matAXC->ax      = (FPT *) calloc( matAXC->lenAX,   sizeof(FPT) ); TEST_POINTER( matAXC->ax );
	matAXC->lenMAPX = (matAXC->lenAX >> 1) + 8;
	matAXC->mapx    = (UIN *) calloc( matAXC->lenMAPX, sizeof(UIN) ); TEST_POINTER( matAXC-> mapx );
	tt = 0.0;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		get_mapxAXC( ompNT, matCSR, matAXC );
		GT( t2 );
		ti = measure_time( t2, t1 );
		tt = tt + ti;
	}
	ti = tt / (double) NUM_ITE;
	tc = tc + ti;
	tt = 0.0;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		get_axAXC( ompNT, matCSR, vec, matAXC );
		GT( t2 );
		ti = measure_time( t2, t1 );
		tt = tt + ti;
	}
	ti = tt / (double) NUM_ITE;
	tc = tc + ti;
	// format's name
	str_formatData fd;
	strcpy( fd.name, "faxc" );
	// AXC memory footprint
	fd.mfp =          (double) ( matAXC->lenAX  * sizeof(FPT) ); // ax
	fd.mfp = fd.mfp + (double) ( matAXC->lenBRP * sizeof(UIN) ); // brp ( stores the starting address of a row )
	// AXC occupancy ( beta )
	fd.beta = ( (double) matAXC->nnz / (double) (matAXC->lenAX >> 1) );
	// AXC conversion time
	fd.ct = tc;
	return( fd );
}

str_res test_gaxc( const UIN cudaBlockSize, const str_matAXC matAXC, const FPT * ref );

static void getArraysLenAXT_UNC_H1( const str_matCSR matCSR, str_matAXT * matAXT )
{
	const UIN nrows = matAXT->nrows;
	const UIN ths   = matAXT->tileHW;
	const UIN ts    = 2 * ths;
	      UIN rid, tiles, totalTiles = 0;
	 matAXT->con[0] = 0;
	for ( rid = 0; rid < nrows; rid++ )
	{
		tiles              = ( matCSR.rl[rid] + ths - 1 ) / ths;
		totalTiles         = totalTiles + tiles;
		matAXT->con[rid+1] = matAXT->con[rid] + tiles * ts;
	}
	matAXT->tileN  = totalTiles;
	matAXT->lenAX  = totalTiles * ts;
	matAXT->lenSEC = totalTiles;
	return;
}



static void getArraysLenAXT_UNC( const str_matCSR matCSR, str_matAXT * matAXT )
{
	const UIN  nrows = matAXT->nrows;
	const UIN    thw = matAXT->tileHW;
	const UIN     th = matAXT->tileH;
	const UIN    ths = thw * th;
	      UIN rid, rowStartPos = 0, tid, fid, cid, positions, totalColumns = 0, totalTiles;
	for ( rid = 0; rid < nrows; rid++ )
	{
		             tid = ( (rowStartPos + ths) / ths ) - 1;
		             fid =    rowStartPos % th;
		             cid = ( (rowStartPos + th) / th ) - 1 - (tid * thw);
		matAXT->con[rid] = tid * (2 * ths) + fid * (2 * thw) + cid;
		       positions = ( ( ( matCSR.rl[rid] + th - 1 ) / th ) * th );
		    totalColumns = totalColumns + ( ( positions + th - 1 ) / th );
		     rowStartPos = rowStartPos + positions;
	}
	totalTiles     = ( totalColumns + thw - 1 ) / thw;
	matAXT->tileN  = totalTiles;
	matAXT->lenAX  = totalTiles * 2 * ths;
	matAXT->lenSEC = totalTiles * thw;
	return;
}



static void getArraysLenAXT_COM_H1( const UIN ompNT, const str_matCSR matCSR, str_matAXT * matAXT )
{
	const UIN nrows = matAXT->nrows;
	const UIN ths   = matAXT->tileHW;
	      UIN rid, totalElements = 0, totalTiles;
	#pragma omp parallel for default(shared) private(rid) reduction(+:totalElements) num_threads(ompNT) schedule(OMP_SCH) if(_OPENMP)
	for ( rid = 0; rid < nrows; rid++ )
		totalElements = totalElements + matCSR.rl[rid];
	totalTiles     = ( totalElements + ths - 1 ) / ths;
	matAXT->tileN  =     totalTiles;
	matAXT->lenAX  = 2 * totalTiles * ths;
	matAXT->lenSEC =     totalTiles * ths;
	return;
}



static void getArraysLenAXT_COM( const UIN ompNT, const str_matCSR matCSR, str_matAXT * matAXT )
{
	const UIN nrows = matAXT->nrows;
	const UIN ths   = matAXT->tileHW * matAXT->tileH;
	      UIN rid, totalElements = 0, totalTiles;
	#pragma omp parallel for default(shared) private(rid) reduction(+:totalElements) num_threads(ompNT) schedule(OMP_SCH) if(_OPENMP)
	for ( rid = 0; rid < nrows; rid++ )
		totalElements = totalElements + matCSR.rl[rid];
	totalTiles     = ( totalElements + ths - 1 ) / ths;
	matAXT->tileN  =     totalTiles;
	matAXT->lenAX  = 2 * totalTiles * ths;
	matAXT->lenSEC =     totalTiles * ths;
	return;
}



static void getArraysAxSecAXT_UNC_H1( const UIN ompNT, const str_matCSR matCSR, const FPT * vec, str_matAXT * matAXT )
{
	const UIN nrows = matAXT->nrows;
	const UIN   thw = matAXT->tileHW;
	const UIN    th = matAXT->tileH;
	const UIN   ths = thw * th;
	      UIN rowID, rowLen, posAX, posSEC, posCSR, ctrEle;
	#pragma omp parallel for default(shared) private(rowID,rowLen,posAX,posSEC,posCSR,ctrEle) num_threads(ompNT) schedule(OMP_SCH) if(_OPENMP)
	for ( rowID = 0; rowID < nrows; rowID++ )
	{
		rowLen = matCSR.rl[rowID];
		if (rowLen>0)
		{
			posAX  = matAXT->con[rowID];
			posSEC = (posAX/(2*ths));
			ctrEle = 0;
			for ( posCSR = matCSR.row[rowID]; posCSR < matCSR.row[rowID+1]; posCSR++ )
			{
				matAXT->ax[posAX]     = matCSR.val[posCSR];
				matAXT->ax[posAX+thw] = vec[matCSR.col[posCSR]];
				matAXT->sec[posSEC]   = rowID;
				posAX++;
				ctrEle++;
				if ((ctrEle%thw)==0)
				{
					posAX = posAX + thw;
					posSEC++;
				}
			}
		}
	}
	return;
}



static void getArraysAxSecAXT_UNC( const UIN ompNT, const str_matCSR matCSR, const FPT * vec, str_matAXT * matAXT )
{
	const UIN nrows = matAXT->nrows;
	const UIN   thw = matAXT->tileHW;
	const UIN    th = matAXT->tileH;
	const UIN   ths = thw * th;
	      UIN rowID, rowLen, posAX, posSEC, posCSR, ctrEle;
	#pragma omp parallel for default(shared) private(rowID,rowLen,posAX,posSEC,posCSR,ctrEle) num_threads(ompNT) schedule(OMP_SCH) if(_OPENMP)
	for ( rowID = 0; rowID < nrows; rowID++ )
	{
		rowLen = matCSR.rl[rowID];
		if (rowLen>0)
		{
			posAX  = matAXT->con[rowID];
			posSEC = (posAX/(2*ths))*thw + posAX%thw;
			ctrEle = 0;
			for ( posCSR = matCSR.row[rowID]; posCSR < matCSR.row[rowID+1]; posCSR++ )
			{
				matAXT->ax[posAX]     = matCSR.val[posCSR];
				matAXT->ax[posAX+thw] = vec[matCSR.col[posCSR]];
				matAXT->sec[posSEC]   = rowID;
				posAX                 = posAX  + 2 * thw;
				ctrEle++;
				if ((ctrEle%th) == 0)
				{
					posAX = posAX + 1 - (2 * th * thw);
					posSEC++;
					if (posAX%thw==0) posAX = posAX + ((2*th)-1) * thw;
				}
			}
		}
	}
	return;
}



static void getArraysAxSecAXT_COM_H1( const UIN bs, const UIN log, const UIN ompNT, const str_matCSR matCSR, const FPT * vec, str_matAXT * matAXT )
{
	const UIN nrows = matAXT->nrows;
	const UIN   thw = matAXT->tileHW;
	      UIN rowID, rowLen, eleCtr, posCSR, bco, tid, tco, posAX, posSEC, q1, q2, offset;
	#pragma omp parallel for default(shared) private(rowID,rowLen,eleCtr,posCSR,bco,tid,tco,posAX,posSEC,q1,q2,offset) num_threads(ompNT) schedule(OMP_SCH) if(_OPENMP)
	for ( rowID = 0; rowID < nrows; rowID++ )
	{
		rowLen = matCSR.rl[rowID];
		if (rowLen>0)
		{
			eleCtr = 0;
			for ( posCSR = matCSR.row[rowID]; posCSR < matCSR.row[rowID+1]; posCSR++ )
			{
				bco                   =   posCSR%bs;
				tid                   = ((posCSR+thw)/thw)-1;
				tco                   =  posCSR%thw;
				posAX                 = tid * 2 * thw + tco;
				posSEC                = tid     * thw + tco;
				matAXT->ax[posAX]     = matCSR.val[posCSR];
				matAXT->ax[posAX+thw] = vec[matCSR.col[posCSR]];
				if ( (eleCtr==0) || (bco==0))
				{
					q1     = rowLen - eleCtr - 1;
					q2     = bs - 1 - bco;
					offset = (q1 > q2) ? q2 : q1;
					matAXT->sec[posSEC] = rowID<<log | offset;
				}
				eleCtr++;
			}
		}
	}
	return;
}



static void getArraysAxSecAXT_COM( const UIN ompNT, str_matCSR matCSR, const FPT * vec, str_matAXT * matAXT )
{
	strcpy( matAXT->name, matCSR.name );
	const UIN nrows = matAXT->nrows;
	const UIN th    = matAXT->tileH;
	const UIN thw   = matAXT->tileHW;
	const UIN log   = matAXT->log;
	const UIN ths   = th * thw;
	const UIN ts    =  2 * ths;
	      UIN rid, rl, ec, pCSR, tid, fid, cid, pAX, pSEC, q1, q2, offset;
	#pragma omp parallel for default(shared) private(rid,rl,ec,pCSR,tid,fid,cid,pAX,pSEC,q1,q2,offset) num_threads(ompNT) schedule(OMP_SCH) if(_OPENMP)
	for ( rid = 0; rid < nrows; rid++ )
	{
		rl = matCSR.rl[rid];
		if (rl>0)
		{
			ec = 0;
			for ( pCSR = matCSR.row[rid]; pCSR < matCSR.row[rid+1]; pCSR++ )
			{
				tid  = ( (pCSR + ths) / ths ) - 1;
				fid  = pCSR % th;
				cid  = ( ( (pCSR - tid * ths) + th ) / th ) - 1;
				pAX  = tid * ts  + 2 * fid * thw + cid;
				pSEC = tid * ths +     fid * thw + cid;
				matAXT->ax[pAX]     = matCSR.val[pCSR];
				matAXT->ax[pAX+thw] = vec[matCSR.col[pCSR]];
				if ( (ec==0) || (fid==0) )
				{
					q1     = rl - ec - 1;
					q2     = th - 1 - fid;
					offset = (q1 > q2) ? q2 : q1;
					matAXT->sec[pSEC] = rid << log | offset;
				}
				ec++;
			}
		}
	}
	return;
}



static str_formatData getFormatDataAXT( const UIN ompNT, const UIN bs, const UIN thw, const UIN th, const char * mode, const str_matCSR matCSR, const FPT * vec, str_matAXT * matAXT )
{
	// set AXT parameters
	strcpy( matAXT->name, matCSR.name );
	matAXT->nrows  = matCSR.nrows;
	matAXT->nnz    = matCSR.nnz;
	matAXT->bs     = bs;
	matAXT->tileHW = thw;
	matAXT->tileH  = th;
	strcpy( matAXT->mode, mode );
	matAXT->lenCON = matCSR.nrows;
	   matAXT->con = (UIN *) calloc( matAXT->lenCON + 1, sizeof(UIN) ); TEST_POINTER( matAXT->con );
	UIN i;
	for ( i = 0; i < 10; i++ )
		if ( ((matAXT->tileH) >> i) == 1 ) matAXT->logTH = i;
	// get AXT arrays' length
	struct timespec t1, t2;
	double ti = 0.0, tt = 0.0, tc = 0.0;
	if (strcmp(mode,"UNC")==0)
	{
		if (th == 1)
		{
			for ( i = 0; i < NUM_ITE; i++ )
			{
				GT( t1 );
				getArraysLenAXT_UNC_H1( matCSR, matAXT );
				GT( t2 );
				ti = measure_time( t2, t1 );
				tt = tt + ti;
			}
		}
		else
		{
			for ( i = 0; i < NUM_ITE; i++ )
			{
				GT( t1 );
				getArraysLenAXT_UNC( matCSR, matAXT );
				GT( t2 );
				ti = measure_time( t2, t1 );
				tt = tt + ti;
			}
		}
	}
	else
	{
		if (th == 1)
		{
			for ( i = 0; i < NUM_ITE; i++ )
			{
				GT( t1 );
				getArraysLenAXT_COM_H1( ompNT, matCSR, matAXT );
				GT( t2 );
				ti = measure_time( t2, t1 );
				tt = tt + ti;
			}
		}
		else
		{
			for ( i = 0; i < NUM_ITE; i++ )
			{
				GT( t1 );
				getArraysLenAXT_COM( ompNT, matCSR, matAXT );
				GT( t2 );
				ti = measure_time( t2, t1 );
				tt = tt + ti;
			}
		}
	}
	ti = tt / (double) NUM_ITE;
	tc = tc + ti;
	// get arrays ax[] and sec[]
	matAXT->ax  = (FPT *) calloc( matAXT->lenAX,  sizeof(FPT) );  TEST_POINTER( matAXT->ax  );
	matAXT->sec = (UIN *) calloc( matAXT->lenSEC, sizeof(UIN) );  TEST_POINTER( matAXT->sec );
	tt = 0.0;
	char buffer[48];
	if (strcmp(mode,"UNC")==0)
	{
		if (th==1)
		{
			for ( i = 0; i < NUM_ITE; i++ )
			{
				GT( t1 );
				getArraysAxSecAXT_UNC_H1( ompNT, matCSR, vec, matAXT );
				GT( t2 );
				ti = measure_time( t2, t1 );
				tt = tt + ti;
			}
			char THW[5];  sprintf( THW,  "%d", thw  );
			strcpy( buffer, "faxtuh1hw" );
			strcat( buffer, THW );
		}
		else
		{
			for ( i = 0; i < NUM_ITE; i++ )
			{
				GT( t1 );
				getArraysAxSecAXT_UNC( ompNT, matCSR, vec, matAXT );
				GT( t2 );
				ti = measure_time( t2, t1 );
				tt = tt + ti;
			}
			char TH[5];   sprintf( TH,   "%d", th   );
			char THW[5];  sprintf( THW,  "%d", thw  );
			strcpy( buffer, "faxtuh" );
			strcat( buffer, TH   );
			strcat( buffer, "hw" );
			strcat( buffer, THW  );
		}
	}
	else
	{
		if (th==1)
		{
			for ( i = 1; i < 11; i++ )
			{
				if ((bs>>i) == 1)
				{
					matAXT->log = i;
					break;
				}
			}
			for ( i = 0; i < NUM_ITE; i++ )
			{
				GT( t1 );
				getArraysAxSecAXT_COM_H1( bs, matAXT->log, ompNT, matCSR, vec, matAXT );
				GT( t2 );
				ti = measure_time( t2, t1 );
				tt = tt + ti;
			}
			char THW[5];  sprintf( THW, "%d", thw );
			char BS[5];   sprintf( BS,  "%d", bs  );
			strcpy( buffer, "faxtch1hw" );
			strcat( buffer, THW  );
			strcat( buffer, "bs" );
			strcat( buffer, BS   );
		}
		else
		{
			for ( i = 1; i < 10; i++ )
			{
				if ((th>>i) == 1)
				{
					matAXT->log = i;
					break;
				}
			}
			for ( i = 0; i < NUM_ITE; i++ )
			{
				GT( t1 );
				getArraysAxSecAXT_COM( ompNT, matCSR, vec, matAXT );
				GT( t2 );
				ti = measure_time( t2, t1 );
				tt = tt + ti;
			}
			char TH[5];  sprintf( TH,  "%d", th  );
			char THW[5]; sprintf( THW, "%d", thw );
			strcpy( buffer, "faxtch" );
			strcat( buffer, TH   );
			strcat( buffer, "hw" );
			strcat( buffer, THW  );
		}
	}
	ti = tt / (double) NUM_ITE;
	tc = tc + ti;
	// AXT specific name
	str_formatData fd;
	strcpy( fd.name, buffer );
	// AXT memory footprint
	fd.mfp =          (double) ( matAXT->lenAX  * sizeof(FPT) ); // ax
	fd.mfp = fd.mfp + (double) ( matAXT->lenSEC * sizeof(UIN) ); // sec
	// AXT occupancy ( beta )
	fd.beta = ( (double) matAXT->nnz / (double) (matAXT->lenAX >> 1) );
	// AXT conversion time
	fd.ct = tc;
	return( fd );
}

str_res test_gaxtuh1hw16w( const UIN wpw, const UIN cbs, const str_matAXT matAXT, const FPT * ref );

str_res test_gaxtuh1hw08w( const UIN wpw, const UIN cbs, const str_matAXT matAXT, const FPT * ref );

str_res test_gaxtuh1hw04w( const UIN wpw, const UIN cbs, const str_matAXT matAXT, const FPT * ref );

str_res test_gaxtuh( const UIN tpw, const UIN cbs, const str_matAXT matAXT, const FPT * ref );

str_res test_gaxtch1( const UIN cudaBlockSize, const str_matAXT matAXT, const FPT * ref );

str_res test_gaxtch( const UIN cudaBlockSize, const str_matAXT matAXT, const FPT * ref );

int main( int argc, char ** argv )
{
	// check input arguments
	str_inputArgs sia = checkArgs( argc, argv );

	// print run settings
	printRunSettings( sia );

	// CSR format  ------------------------------------------------------------------------------------------------------------------
	// read matrix in CSR format
	str_matCSR matCSR = matrixReading( sia.matFileName );
	// print matrix's statistics
	printMatrixStats( sia.matFileName, &matCSR );

	// get memory footprint, occupancy (beta) and conversion time
	str_formatData fd01 = getFormatDataCSR( &matCSR );

	// CSR format  ------------------------------------------------------------------------------------------------------------------
	// init vectors to perform SpMV multiplication and check errors (spM * vr = yr)
	FPT * vr = (FPT *) calloc( matCSR.nrows, sizeof(FPT) ); TEST_POINTER( vr );
	init_vec( sia.ompMT, matCSR.nrows, vr );
	FPT * yr = (FPT *) calloc( matCSR.nrows,  sizeof(FPT) ); TEST_POINTER( yr );
	ncsr( sia.ompMT, matCSR, vr, yr );
	
	std::vector<str_res> srs;

#define STR_RES(...) srs.push_back(__VA_ARGS__)
	
	// test CSR kernels
	STR_RES(test_ncsr( sia.ompMT, matCSR, vr, yr ));
	STR_RES(test_gcsr( sia.cbs, matCSR, vr, yr ));
	STR_RES(test_gcucsr( matCSR, vr, yr ));
	// CSR format  ------------------------------------------------------------------------------------------------------------------

	// K1 format  -------------------------------------------------------------------------------------------------------------------
	str_matK1 matK1; str_formatData fd02 = getFormatDataK1( CHUNK_SIZE, matCSR, vr, &matK1 );
	STR_RES(test_gk1( sia.cbs, matK1, vr, yr ));
	// K1 format  -------------------------------------------------------------------------------------------------------------------

	// AXC format  ------------------------------------------------------------------------------------------------------------------
	str_matAXC matAXC; str_formatData fd03 = getFormatDataAXC( sia.ompMT, TILE_HW, matCSR, vr, &matAXC );
	STR_RES(test_gaxc( sia.cbs, matAXC, yr ));
	// AXC format  ------------------------------------------------------------------------------------------------------------------

	// AXT format  ------------------------------------------------------------------------------------------------------------------
	str_matAXT matAXT01; str_formatData fd04 = getFormatDataAXT( sia.ompMT, sia.cbs,      16,  1, "UNC", matCSR, vr, &matAXT01 );
	str_matAXT matAXT02; str_formatData fd05 = getFormatDataAXT( sia.ompMT, sia.cbs,       8,  1, "UNC", matCSR, vr, &matAXT02 );
	str_matAXT matAXT03; str_formatData fd06 = getFormatDataAXT( sia.ompMT, sia.cbs,       4,  1, "UNC", matCSR, vr, &matAXT03 );
	str_matAXT matAXT04; str_formatData fd07 = getFormatDataAXT( sia.ompMT, sia.cbs, TILE_HW,  4, "UNC", matCSR, vr, &matAXT04 );
	str_matAXT matAXT05; str_formatData fd08 = getFormatDataAXT( sia.ompMT, sia.cbs, TILE_HW,  6, "UNC", matCSR, vr, &matAXT05 );
	str_matAXT matAXT06; str_formatData fd09 = getFormatDataAXT( sia.ompMT, sia.cbs, TILE_HW,  8, "UNC", matCSR, vr, &matAXT06 );
	str_matAXT matAXT07; str_formatData fd10 = getFormatDataAXT( sia.ompMT, sia.cbs, TILE_HW, 10, "UNC", matCSR, vr, &matAXT07 );
	str_matAXT matAXT08; str_formatData fd11 = getFormatDataAXT( sia.ompMT, sia.cbs, TILE_HW, 12, "UNC", matCSR, vr, &matAXT08 );
	str_matAXT matAXT09; str_formatData fd12 = getFormatDataAXT( sia.ompMT, sia.cbs, TILE_HW,  1, "COM", matCSR, vr, &matAXT09 );
	str_matAXT matAXT10; str_formatData fd13 = getFormatDataAXT( sia.ompMT, sia.cbs, TILE_HW,  4, "COM", matCSR, vr, &matAXT10 );
	str_matAXT matAXT11; str_formatData fd14 = getFormatDataAXT( sia.ompMT, sia.cbs, TILE_HW,  8, "COM", matCSR, vr, &matAXT11 );
	str_matAXT matAXT12; str_formatData fd15 = getFormatDataAXT( sia.ompMT, sia.cbs, TILE_HW, 16, "COM", matCSR, vr, &matAXT12 );
	STR_RES(test_gaxtuh1hw16w(  64, sia.cbs, matAXT01, yr ));
	STR_RES(test_gaxtuh1hw16w( 128, sia.cbs, matAXT01, yr ));
	STR_RES(test_gaxtuh1hw16w( 192, sia.cbs, matAXT01, yr ));
	STR_RES(test_gaxtuh1hw16w( 256, sia.cbs, matAXT01, yr ));
	STR_RES(test_gaxtuh1hw16w( 320, sia.cbs, matAXT01, yr ));
	STR_RES(test_gaxtuh1hw16w( 384, sia.cbs, matAXT01, yr ));
	STR_RES(test_gaxtuh1hw16w( 448, sia.cbs, matAXT01, yr ));
	STR_RES(test_gaxtuh1hw16w( 512, sia.cbs, matAXT01, yr ));

	STR_RES(test_gaxtuh1hw08w(  64, sia.cbs, matAXT02, yr ));
	STR_RES(test_gaxtuh1hw08w( 128, sia.cbs, matAXT02, yr ));
	STR_RES(test_gaxtuh1hw08w( 192, sia.cbs, matAXT02, yr ));
	STR_RES(test_gaxtuh1hw08w( 256, sia.cbs, matAXT02, yr ));
	STR_RES(test_gaxtuh1hw08w( 320, sia.cbs, matAXT02, yr ));
	STR_RES(test_gaxtuh1hw08w( 384, sia.cbs, matAXT02, yr ));
	STR_RES(test_gaxtuh1hw08w( 448, sia.cbs, matAXT02, yr ));
	STR_RES(test_gaxtuh1hw08w( 512, sia.cbs, matAXT02, yr ));

	STR_RES(test_gaxtuh1hw04w(  64, sia.cbs, matAXT03, yr ));
	STR_RES(test_gaxtuh1hw04w( 128, sia.cbs, matAXT03, yr ));
	STR_RES(test_gaxtuh1hw04w( 192, sia.cbs, matAXT03, yr ));
	STR_RES(test_gaxtuh1hw04w( 256, sia.cbs, matAXT03, yr ));
	STR_RES(test_gaxtuh1hw04w( 320, sia.cbs, matAXT03, yr ));
	STR_RES(test_gaxtuh1hw04w( 384, sia.cbs, matAXT03, yr ));
	STR_RES(test_gaxtuh1hw04w( 448, sia.cbs, matAXT03, yr ));
	STR_RES(test_gaxtuh1hw04w( 512, sia.cbs, matAXT03, yr ));

	STR_RES(test_gaxtuh     ( 1, sia.cbs, matAXT04, yr ));
	STR_RES(test_gaxtuh     ( 2, sia.cbs, matAXT04, yr ));
	STR_RES(test_gaxtuh     ( 3, sia.cbs, matAXT04, yr ));
	STR_RES(test_gaxtuh     ( 4, sia.cbs, matAXT04, yr ));

	STR_RES(test_gaxtuh     ( 1, sia.cbs, matAXT05, yr ));
	STR_RES(test_gaxtuh     ( 2, sia.cbs, matAXT05, yr ));
	STR_RES(test_gaxtuh     ( 3, sia.cbs, matAXT05, yr ));
	STR_RES(test_gaxtuh     ( 4, sia.cbs, matAXT05, yr ));

	STR_RES(test_gaxtuh     ( 1, sia.cbs, matAXT06, yr ));
	STR_RES(test_gaxtuh     ( 2, sia.cbs, matAXT06, yr ));
	STR_RES(test_gaxtuh     ( 3, sia.cbs, matAXT06, yr ));
	STR_RES(test_gaxtuh     ( 4, sia.cbs, matAXT06, yr ));

	STR_RES(test_gaxtuh     ( 1, sia.cbs, matAXT07, yr ));
	STR_RES(test_gaxtuh     ( 2, sia.cbs, matAXT07, yr ));
	STR_RES(test_gaxtuh     ( 3, sia.cbs, matAXT07, yr ));
	STR_RES(test_gaxtuh     ( 4, sia.cbs, matAXT07, yr ));

	STR_RES(test_gaxtuh     ( 1, sia.cbs, matAXT08, yr ));
	STR_RES(test_gaxtuh     ( 2, sia.cbs, matAXT08, yr ));
	STR_RES(test_gaxtuh     ( 3, sia.cbs, matAXT08, yr ));
	STR_RES(test_gaxtuh     ( 4, sia.cbs, matAXT08, yr ));

	STR_RES(test_gaxtch1    ( sia.cbs, matAXT09, yr ));

	STR_RES(test_gaxtch     ( sia.cbs, matAXT10, yr ));
	STR_RES(test_gaxtch     ( sia.cbs, matAXT11, yr ));
	STR_RES(test_gaxtch     ( sia.cbs, matAXT12, yr ));
	// AXT format  ------------------------------------------------------------------------------------------------------------------

	HDL; printf( "formats' data\n" ); HDL;
	printf( "%25s %20s %10s %20s\n", "format", "memory [Mbytes]", "occupancy", "convTime [s]" );
	printf( "%25s %20.2lf %10.2lf %20.6lf\n", fd01.name, ( fd01.mfp * 1e-6 ), fd01.beta, fd01.ct ); fflush(stdout);
	printf( "%25s %20.2lf %10.2lf %20.6lf\n", fd02.name, ( fd02.mfp * 1e-6 ), fd02.beta, fd02.ct ); fflush(stdout);
	printf( "%25s %20.2lf %10.2lf %20.6lf\n", fd03.name, ( fd03.mfp * 1e-6 ), fd03.beta, fd03.ct ); fflush(stdout);
	printf( "%25s %20.2lf %10.2lf %20.6lf\n", fd04.name, ( fd04.mfp * 1e-6 ), fd04.beta, fd04.ct ); fflush(stdout);
	printf( "%25s %20.2lf %10.2lf %20.6lf\n", fd05.name, ( fd05.mfp * 1e-6 ), fd05.beta, fd05.ct ); fflush(stdout);
	printf( "%25s %20.2lf %10.2lf %20.6lf\n", fd06.name, ( fd06.mfp * 1e-6 ), fd06.beta, fd06.ct ); fflush(stdout);
	printf( "%25s %20.2lf %10.2lf %20.6lf\n", fd07.name, ( fd07.mfp * 1e-6 ), fd07.beta, fd07.ct ); fflush(stdout);
	printf( "%25s %20.2lf %10.2lf %20.6lf\n", fd08.name, ( fd08.mfp * 1e-6 ), fd08.beta, fd08.ct ); fflush(stdout);
	printf( "%25s %20.2lf %10.2lf %20.6lf\n", fd09.name, ( fd09.mfp * 1e-6 ), fd09.beta, fd09.ct ); fflush(stdout);
	printf( "%25s %20.2lf %10.2lf %20.6lf\n", fd10.name, ( fd10.mfp * 1e-6 ), fd10.beta, fd10.ct ); fflush(stdout);
	printf( "%25s %20.2lf %10.2lf %20.6lf\n", fd11.name, ( fd11.mfp * 1e-6 ), fd11.beta, fd11.ct ); fflush(stdout);
	printf( "%25s %20.2lf %10.2lf %20.6lf\n", fd12.name, ( fd12.mfp * 1e-6 ), fd12.beta, fd12.ct ); fflush(stdout);
	printf( "%25s %20.2lf %10.2lf %20.6lf\n", fd13.name, ( fd13.mfp * 1e-6 ), fd13.beta, fd13.ct ); fflush(stdout);
	printf( "%25s %20.2lf %10.2lf %20.6lf\n", fd14.name, ( fd14.mfp * 1e-6 ), fd14.beta, fd14.ct ); fflush(stdout);
	printf( "%25s %20.2lf %10.2lf %20.6lf\n", fd15.name, ( fd15.mfp * 1e-6 ), fd15.beta, fd15.ct ); fflush(stdout);

	// Mark kernel with best performance with an asterisk.
	std::map<double, str_res> sorted_res;
	for (auto sr : srs)
		sorted_res[sr.flops] = sr;
	auto& best_res = sorted_res.rbegin()->second;
	best_res.name[strlen(best_res.name) + 1] = '\0';
       	best_res.name[strlen(best_res.name)] = '*';

	HDL; printf( "SpMV kernels' results\n" ); HDL;
	printf( "%25s %15s %8s %15s %13s %13s %10s\n", "kernel", "exeTime [s]", "Gflops", "ordTime [s]", "errAbs", "errRel", "rowInd" );
	for (auto i = sorted_res.rbegin(), e = sorted_res.rend(); i != e; i++)
	{
		const auto& sr = i->second;
		printf( "%25s %15.7lf %8.3lf %15.7lf %11.3le %13.3le %12d\n",
			sr.name, sr.et, ( sr.flops * 1e-9 ), sr.ot,
			sr.sErr.aErr, sr.sErr.rErr, sr.sErr.pos );
		fflush(stdout);
	}

	return( EXIT_SUCCESS );
}


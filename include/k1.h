#ifndef LIBAXT_K1_H
#define LIBAXT_K1_H

typedef struct { UIN nrows; UIN nnz; UIN chunkNum; UIN lenVC; UIN * permi; UIN * nmc; UIN * chp; FPT * val; UIN * col; } str_matK1;

#endif // LIBAXT_K1_H


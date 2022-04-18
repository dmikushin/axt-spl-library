#ifndef LIBAXT_AXT_H
#define LIBAXT_AXT_H

typedef struct{ char name[48]; UIN nrows; UIN nnz; char mode[8]; UIN tileHW; UIN tileH; UIN logTH; UIN tileN; UIN lenAX; UIN lenSEC; UIN lenCON; UIN log; UIN bs; FPT * ax; UIN * sec; UIN * con; } str_matAXT;

#endif // LIBAXT_AXT_H


#ifndef LIBAXT_NCSR_H
#define LIBAXT_NCSR_H

static void ncsr( const UIN ompNT, const str_matCSR matCSR, const FPT * vec, FPT * res )
{
        UIN i, j;
        FPT aux;
        #pragma omp parallel for default(shared) private(i,j,aux) num_threads(ompNT) schedule(OMP_SCH) if(_OPENMP)
        for ( i = 0; i < matCSR.nrows; i++ )
        {
                aux = (FPT) 0;
                for ( j = matCSR.row[i]; j < matCSR.row[i+1]; j++ )
                {
                        aux = aux + matCSR.val[j] * vec[matCSR.col[j]];
                }
                res[i] = aux;
        }
        return;
}

#endif // LIBAXT_NCSR_H


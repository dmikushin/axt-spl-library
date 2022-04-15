#ifndef AXT_DIFF_H
#define AXT_DIFF_H

static void get_errors( const UIN len, const FPT * ar, const FPT * ac, str_err * sErr )
{
        double dif, maxDif = 0.0;
        double val, maxVal = 0.0;
        UIN pos = 0;
        UIN i;
        for ( i = 0; i < len; i++ )
        {
                val = fabs(ar[i]);
                if ( val > maxVal ) maxVal = val;
                dif = fabs( fabs(ac[i]) - val );
                if ( dif > maxDif )
                {
                        maxDif = dif;
                        pos    = i;
                }
        }
        sErr->aErr = maxDif;
        sErr->rErr = maxDif/maxVal;
        sErr->pos  = pos;
        return;
}

#endif // AXT_DIFF_H


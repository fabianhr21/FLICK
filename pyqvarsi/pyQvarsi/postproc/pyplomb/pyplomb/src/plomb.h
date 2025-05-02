/*
    LOMB-SCARGLE PERIODOGRAM
*/

#define PI   3.141592653589793115997963468544185161590576171875000
#define PI2  6.28318530717958623199592693708837032318115234375000
#define MACC 4

#define LMIN(a,b) (lminarg1=(a),lminarg2=(b),(lminarg1) < (lminarg2) ? (lminarg1) : (lminarg2))
#define LMAX(a,b) (lmaxarg1=(a),lmaxarg2=(b),(lmaxarg1) > (lmaxarg2) ? (lmaxarg1) : (lmaxarg2))

#define MOD(a,b)  while(a >= b) a -= b;
#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))
#define X2(a)     ((a)*(a))
#define SWAP(a,b) tmp=(a);(a)=(b);(b)=tmp

int nextPowerOf2(int p, int n);

int splomb(float  *x, float  *y, const int n, float  o, float  hi, float  *w1, float  *w2, const int nw);
int dplomb(double *x, double *y, const int n, double o, double hi, double *w1, double *w2, const int nw);

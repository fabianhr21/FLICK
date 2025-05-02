/*
    LOMB-SCARGLE PERIODOGRAM

    Single precision version
*/
#include <math.h>
#include "plomb.h"

static long lmaxarg1,lmaxarg2;
static long lminarg1,lminarg2;


void smeanvarrange(float *ymean, float *yvar, float *xmax, float *xmin, float *y, float *x, const int n) {
    /*
        mean, variance and range using Welford's online algorithm
    */
    // Initialize
    float ymean_old = 0.;
    *ymean = 0.; *yvar = 0.; *xmax = -1.e99; *xmin = 1.e99;
    
    for (int i=0; i<n; ++i) {
        // Welford oneline for mean and variance
        ymean_old = *ymean;
        *ymean    = ymean_old + (1./(float)(i+1)) * (y[i] - ymean_old);
        *yvar    += (y[i] - ymean_old)*(y[i] - *ymean);
        // Maximum and minimum
        *xmin     = fmin(*xmin,x[i]);
        *xmax     = fmax(*xmax,x[i]);
    }
    *yvar /= (float)(n-1);
}

void sspread(float y, float *yy, int n, float x, int m) {
    /*
        Spread an array yy extrapolating the value using the 
        Lagrange interpolating polynomial.
    */
    static long lagr[11]={0,1,1,2,6,24,120,720,5040,40320,362880};
    int ilow, ihigh, nden;
    int ix = (int)(x);
    float fac;

    if ( x == (float)(ix) ) {
        yy[ix] += y;
    } else {
        ilow  = LMIN( LMAX((long)(x - 0.5*m + 1.),1) , n - m + 1);
        ihigh = ilow + m - 1;
        nden  = lagr[m];
        fac   = x - ilow;
        for (int i=ilow+1;i<=ihigh;++i) fac *= (x - i);
        
        yy[ihigh] += y*fac/(nden*(x - ihigh));

        for (int i=ihigh-1;i>=ilow;--i) {
            nden   = (nden/(i + 1 - ilow))*(i - ihigh);
            yy[i] += y*fac/(nden*(x - i));
        }
    }
}

void sdft1(float *x, int nn, int isign) {
    /*
        Discrete Fourier Transform
    */
    float tmp, tmpr, tmpi, theta;
    float wpr, wpi, wr, wi;
    int i, j = 1, n = nn << 1, m, mmax, istep;

    for (i=1; i<n; i+=2) {
        if ( j > i) {
            SWAP(x[j],x[i]);
            SWAP(x[j+1],x[i+1]);
        }

        m = nn;
        while (m >= 2 && j > m) { j -= m; m >>= 1; }
        j += m;
    }

	mmax = 2;
	while(n > mmax){
		istep = mmax << 1;
		theta = isign*(PI2/mmax);
		tmp   = sin(0.5*theta);
		wpr   = -2.0*X2(tmp);
		wpi   = sin(theta);
		wr    = 1.0;
		wi    = 0.0;

		for(m=1; m<mmax; m+=2){
			for(i=m; i<=n; i+=istep){
				j       = i + mmax;
				tmpr    = wr*x[j]   - wi*x[j+1];
				tmpi    = wr*x[j+1] + wi*x[j];
				x[j]    = x[i]   - tmpr;
				x[j+1]  = x[i+1] - tmpi;
				x[i]   += tmpr;
				x[i+1] += tmpi;
			}

		wr = (tmp = wr)*wpr - wi*wpi + wr;
		wi = wi*wpr + tmp*wpi + wi;
		}
		mmax = istep;
    }
}

void srfft(float *x, int n, int isign) {
    /*
        Compute the real Fast Fourier transform
    */    
    float c1 = 0.5, c2;
    float theta = PI/(float)(n>>1);

    if (isign == 1) {
        // Forward transform
        c2 = -0.5;
        sdft1(x,n>>1,1);
    } else {
        // Inverse transform
        c2    = 0.5;
        theta = -theta;
    }

    float wtmp = sin(0.5*theta);
    float wpr  = -2.*wtmp*wtmp;
    float wpi  = sin(theta);
    float wr   = 1. + wpr;
    float wi   = wpi;

    float h1r,h1i,h2r,h2i;
    int i1, i2, i3, i4, n3 = n + 3;
    for (int i=2; i<=(n>>2); ++i) {
        i4  = 1 + (i3 = n3 - (i2 = 1 + (i1 = i + i - 1)));
        // The two separate transforms are separated out of data
        h1r =  c1*(x[i1] + x[i3]); 
        h1i =  c1*(x[i2] - x[i4]);
        h2r = -c2*(x[i2] + x[i4]);
        h2i =  c2*(x[i1] - x[i3]);
        // Here they are recombined to form the true transform of 
        // the original real data
        x[i1] =  h1r + wr*h2r - wi*h2i; 
        x[i2] =  h1i + wr*h2i + wi*h2r;
        x[i3] =  h1r - wr*h2r + wi*h2i;
        x[i4] = -h1i + wr*h2i + wi*h2r;
        //The recurrence
        wr = (wtmp = wr)*wpr - wi*wpi + wr; 
        wi = wi*wpr + wtmp*wpi + wi;
    }

    if (isign == 1) {
        x[1] = (h1r=x[1]) + x[2];
        x[2] = h1r - x[2];
    } else {
        x[1] = c1*((h1r = x[1]) + x[2]);
        x[2] = c1*(h1r - x[2]);
        sdft1(x,n>>1,-1);
    }
}

int splomb(float *x, float *y, const int n, float o, float hi, float *w1, float *w2, const int nw) {
    /*
        LOMB-SCARGLE periodogram
    */
    // Compute the FFT size as the next power of 2
    int nout = 0.5*o*hi*n;
    int ndim = nextPowerOf2(64,o*hi*n*MACC) << 1;
    
    // Compute the mean, variance and range of the data
    float ymean, yvar, xmin, xmax, xdiff;
    smeanvarrange(&ymean,&yvar,&xmax,&xmin,y,x,n);
    xdiff = xmax - xmin;

    // Zero-out the workspaces
    for (int i=0; i<=ndim; ++i) w1[i] = w2[i] = 0.;
    float fac = ndim/(xdiff*o), fndim = ndim;

    // Extrapolate the data into the workspaces
    float xk, xkk;
    for (int i=1; i<=n; ++i) {
        xk  = ((x[i] - xmin)*fac);
        MOD(xk,fndim);
        xkk = 2.*(xk++);
        MOD(xkk,fndim);
        ++xkk;
        sspread(y[i] - ymean,w1,ndim,xk,MACC);
        sspread(1.0,w2,ndim,xkk,MACC);
    }

    // Compute the real Fast Fourier Transforms
	srfft(w1,ndim,1); srfft(w2,ndim,1);
	float df = 1./(xdiff*o);

	// Compute the Lomb value for each frequency
	float cterm, cwt, den, hc2wt, hs2wt, hypo, sterm, swt;

	for (int k=3,j=1; j<=(nout); ++j,k+=2) {
		hypo  = sqrt(X2(w2[k] + X2(w2[k+1])));
        hc2wt = 0.5*w2[k]/hypo;
		hs2wt = 0.5*w2[k+1]/hypo;
		cwt   = sqrt(0.5 + hc2wt);
		swt   = SIGN(sqrt(0.5 - hc2wt),hs2wt);
		den   = 0.5*n + hc2wt*w2[k] + hs2wt*w2[k+1];
		cterm = X2(cwt*w1[k]   + swt*w1[k+1])/den;
		sterm = X2(cwt*w1[k+1] - swt*w1[k])/(n - den);
		w1[j] = j*df;
		w2[j] = (cterm + sterm)/(2.0*yvar);
	}

    return nout;
}
/*
    LOMB-SCARGLE PERIODOGRAM

    Single precision version
*/
#include "plomb.h"


int nextPowerOf2(int p, int n) {
	if (n && !(n & (n - 1))) return n;
	while (p < n) p <<= 1;
	return p;
}
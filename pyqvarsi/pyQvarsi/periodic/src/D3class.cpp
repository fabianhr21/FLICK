#include "D3class.h"

using namespace std;


D3::D3() {
	gzero=1.0e-10;
	pd[0]=0; pd[1]=0; pd[2]=0;        
}

D3::D3(const double x,const double y,const double z) {
	gzero=1.0e-10;
	pd[0]=x; pd[1]=y; pd[2]=z;
}

D3::D3(const double x,const double y,const double z,double gzero) {
	gzero=gzero;
	pd[0]=x; pd[1]=y; pd[2]=z;
}

D3::D3(const double* x) {
	gzero=1.0e-10;
	pd[0]=x[0]; pd[1]=x[1]; pd[2]=x[2];
}

D3::D3(const D3 &p)
{
	gzero=1.0e-10;
	pd[0]=p.pd[0]; 
	pd[1]=p.pd[1]; 
	pd[2]=p.pd[2];
}

D3::~D3() {
}



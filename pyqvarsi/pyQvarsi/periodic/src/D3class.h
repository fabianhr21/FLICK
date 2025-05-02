#ifndef D3_CLASS
#define D3_CLASS

#include <iostream>
#include <math.h>
#include <stdlib.h>

using namespace std;
class D3 { 

   friend inline D3 operator*(double l, const D3 &a);
   friend ostream &operator<<(ostream &, const D3 &);

    public:

    D3();
    D3(const double x,const double y,const double z);
    D3(const double x,const double y,const double z,double gzero);
    D3(const D3 &p);
    D3(const double*);
    ~D3();

    inline double norm2() const;                  

    inline D3 operator-(const double &a) const;  
    inline D3 operator+(const double &a) const; 
    inline D3 operator-(const D3 &a) const;  
    inline D3 operator+(const D3 &a) const;  
    inline D3 operator^(const D3 &b) const;
    inline D3 operator*(const double l) const; 
    inline double operator*(const D3 &a) const;
    inline D3& operator=(const D3 &a);  
    inline void operator+=(const D3 &);       
    inline void operator-=(const D3 &);
    inline void operator*=(const double l);
    inline bool operator==(const D3 &) const; 
    inline bool operator<(const D3 &) const;  
    inline double  operator[](int i) const;
    inline double& operator[](int i);     

    private:   
    double pd[3];
    double gzero =1e-10;	
};

inline D3 D3::operator-(const double &a) const { 
    return ( D3 ( pd[0] - a , pd[1] - a , pd[2] - a ) );
}

inline D3 D3::operator+(const double &a) const { 
    return ( D3 ( pd[0] + a , pd[1] + a , pd[2] + a ) );
}

inline D3 D3::operator-(const D3 &a) const { 
    return ( D3 ( pd[0] - a.pd[0] , pd[1] - a.pd[1] , pd[2] - a.pd[2] ) );
}

inline D3 D3::operator+(const D3 &a) const { 
    return ( D3 ( pd[0] + a.pd[0] , pd[1] + a.pd[1] , pd[2] + a.pd[2] ) );
}

inline D3 D3::operator*(const double l) const {
    return ( D3 ( pd[0]*l , pd[1]*l , pd[2]*l ) );
}
inline double D3::operator*(const D3 &a) const {
    return ( pd[0]*a.pd[0] + pd[1]*a.pd[1] + pd[2]*a.pd[2] );
}

inline void D3::operator+=(const D3 &a){  
    pd[0]+=a.pd[0];
    pd[1]+=a.pd[1];
    pd[2]+=a.pd[2];

}
inline void D3::operator*=(const double l)
{
    pd[0]*=l;
    pd[1]*=l;
    pd[2]*=l;
}


inline void D3::operator-=(const D3 &a){ 
    pd[0]-=a.pd[0];
    pd[1]-=a.pd[1];
    pd[2]-=a.pd[2];

}
inline bool D3::operator==(const D3 &a) const{  
    return (((*this)-a).norm2()<gzero) ? true : false;
}
inline bool D3::operator<(const D3 &a)const{  
    if((pd[0]-a.pd[0])>gzero) return true;
    else{ 
        if(fabs(pd[0]-a.pd[0])<gzero){
            if((a.pd[1]-pd[1])>gzero) return true;
            else{
                if(fabs(pd[1]-a.pd[1])<gzero){
                    if((a.pd[2]-pd[2])>gzero) return true;
                    else return false;
                }
                else return false;
            }
        }
        else return false;
    }
}
inline double D3::operator[](int i) const{ 
    return pd[i];
}

inline double& D3::operator[](int i){ 
    return pd[i];
}

inline D3& D3::operator=(const D3 &a) { 
    pd[0]=a.pd[0]; 
    pd[1]=a.pd[1]; 
    pd[2]=a.pd[2]; 
    return *this; 
}

inline double D3::norm2() const {return sqrt((pd[0]*pd[0]+pd[1]*pd[1]+pd[2]*pd[2]));}


inline D3  D3::operator^(const D3 &b) const  
{ 
    D3 r(pd[1]*b.pd[2] - pd[2]*b.pd[1] ,
            -pd[0]*b.pd[2] + pd[2]*b.pd[0] , 
            pd[0]*b.pd[1] - pd[1]*b.pd[0]);

    return r;                              
}

inline D3 operator*(double l, const D3 &a)      
{
    D3 r(l*a.pd[0],l*a.pd[1], l*a.pd[2]); 
    return r;                               
}
#endif


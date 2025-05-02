/* 
	GEOMETRY
	
	Definition of a generic geometric tools such as polygons.

	Arnau Miro (OGS) (c) 2019
*/

#ifndef Geometry_h
#define Geometry_h

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cmath>

namespace Geom
{
	class Point;
	class Vector;
	class Polygon;

	int cn_PinPoly(const Polygon *poly, const Point &P); // Return:  0 = outside, 1 = inside
	int wn_PinPoly(const Polygon *poly, const Point &P); // Return:  =0 only when P is outside


	class Point {

		public:
			// Constructors and destructors
			inline Point()                                                     { set(0,0,0); }
			inline Point(const double x, const double y, const double z)       { set(x,y,z); }
			inline Point(const double *v)                                      { set(v[0],v[1],v[2]); }
			inline Point(const Point &pp)                                      { set(pp[0],pp[1],pp[2]); }
			inline ~Point()                                                    {}

			// Functions
			inline void    set(const double x, const double y, const double z) { p[0] = x; p[1] = y; p[2] = z; }
			inline void    set(const double *v)                                { set(v[0],v[1],v[2]); }
			inline void    set(const Point &pp)                                { set(pp[0],pp[1],pp[2]); }
			inline void    set(const int i, const double val)                  { p[i] = val; }
			inline double  get(const int i)                                    { return p[i]; }
			inline double *data()                                              { return p; }
			inline double  x()                                                 { return p[0]; }
			inline double  y()                                                 { return p[1]; }
			inline double  z()                                                 { return p[2]; }
			inline void    print() const                                       { std::printf("[ %f %f %f ]",p[0],p[1],p[2]); }

			inline double  dist(const Point &pp) const;
			inline double  dist2(const Point &pp) const;

			/* ISLEFT

				Tests if a point is Left|On|Right of an infinite line.

				Input:  two points P0, P1; defining a line
				Return: >0 for P2 left of the line through P0 and P1
				        =0 for P2  on the line
				        <0 for P2  right of the line
				
				from: http://geomalgorithms.com/a03-_inclusion.html
				
				Copyright 2001 softSurfer, 2012 Dan Sunday
				This code may be freely used and modified for any purpose
				providing that this copyright notice is included with it.
				SoftSurfer makes no warranty for this code, and cannot be held
				liable for any real or imagined damage resulting from its use.
				Users of this code must verify correctness for their application.
			*/
			inline double isLeft(const Point &P0, const Point &P1) const { return ( (P1[0] - P0[0])*(p[1] - P0[1]) - (p[0]-P0[0])*(P1[1]-P0[1]) ); }

			// Operators
			inline double  operator[](int i) const           { return p[i]; }
			inline double &operator[](int i)                 { return p[i]; }
			inline Point  &operator=(const Point &pp);
			inline Point   operator+(const Vector &vv) const;
			inline Point   operator-(const Vector &vv) const;
			inline Vector  operator-(const Point &pp) const;
			inline bool    operator==(const Point &v) const;
			inline bool    operator!=(const Point &v) const;

		private:
			double p[3];
	};

	class Vector {

		public:
			// Constructors and destructors
			inline Vector()                                                    { set(0,0,0); }
			inline Vector(const double x, const double y, const double z)      { set(x,y,z); }
			inline Vector(const double *p)                                     { set(p[0],p[1],p[2]); }
			inline Vector(const Vector &vv)                                    { set(vv[0],vv[1],vv[2]); }
			inline ~Vector()                                                   {}

			// Functions
			inline void    set(const double x, const double y, const double z) { v[0] = x; v[1] = y; v[2] = z; }
			inline void    set(const double *p)                                { set(p[0],p[1],p[2]); }
			inline void    set(const Vector &vv)                               { set(vv[0],vv[1],vv[2]); }
			inline void    set(const int i, const double val)                  { v[i] = val; }
			inline double  get(const int i)                                    { return v[i]; }			
			inline double *data()                                              { return v; }
			inline double  x()                                                 { return v[0]; }
			inline double  y()                                                 { return v[1]; }
			inline double  z()                                                 { return v[2]; }
			
			inline double  dot(const Vector &vv) const                         { return ( (double)(v[0]*vv [0] + v[1]*vv [1] + v[2]*vv [2]) ); }
			inline Vector  cross(const Vector &vv) const                       { return ( Vector(v[1]*vv[2]-v[2]*vv[1], -v[0]*vv[2]+v[2]*vv[0], v[0]*vv[1]-v[1]*vv[0]) ); }
			inline double  norm() const                                        { return ( (double)(std::sqrt(norm2())) ); }
			inline double  norm2() const                                       { return ( dot((*this)) ); }
			inline void    print() const                                       { std::printf("( %f %f %f )",v[0],v[1],v[2]); }

			// Operators
			inline double  operator[](int i) const                             { return v[i]; }
			inline double &operator[](int i)                                   { return v[i]; }
			inline Vector &operator=(const Vector &vv);
			inline Vector  operator+(const Vector &vv) const;
			inline Vector  operator-(const Vector &vv) const;
			inline Vector  operator*(const double a) const;
			inline Vector  operator/(const double a) const;
			inline double  operator*(const Vector &vv) const;
			inline Vector  operator^(const Vector &vv) const;

		private:
			double v[3];
	};


	class Ball {

		public:
			// Constructor and destructors
			inline Ball()                                     { set(Point(0,0,0),0); }
			inline Ball(const double r, const Point &p)       { set(r,p); }
			inline Ball(const Point &p, const double r)       { set(r,p); }
			inline Ball(const Polygon &p)                     { fastBall(p); }
			inline Ball(const Ball &b)                        { set(b.get_center(),b.get_radius()); }
			inline ~Ball()                                    {}

			// Functions
			inline void   set(const double r, const Point &p) { set_center(p); set_radius(r); }
			inline void   set(const Point &p, const double r) { set_center(p); set_radius(r); }
			inline void   set_center(const Point &p)          { center = p; }
			inline void   set_radius(const double r)          { radius = r; }
			inline Point  get_center() const                  { return center; }
			inline double get_radius() const                  { return radius; }

			inline void   print() const                       { std::printf("center = "); center.print(); printf(" radius = %f\n",radius); }

			inline bool   isempty() const                     { return ( radius == 0 ); }
			inline bool   isinside(const Point &p) const      { return ( (!isempty() && p.dist(center) < (double)(radius)) ? true : false ); }
			inline bool   isdisjoint(const Ball &b) const     { return ( (!isempty() && b.get_center().dist(center) < (double)(radius + b.get_radius())) ? true : false ); }

			void   fastBall(const Polygon &p);

			// Operators
			inline Ball  &operator=(const Ball &b)            { set(b.get_center(),b.get_radius()); return (*this); }
			inline bool   operator==(const Ball &b) const     { return (center == b.get_center() && radius == b.get_radius()); }
			inline bool   operator>(const Point &p) const     { return ( isinside(p) ); }   // Return true if Point inside Ball
			inline bool   operator<(const Point &p) const     { return ( !isinside(p) ); }  // Return true if Point outside Ball

		private:
			Point center;
			double radius;
	};


	class Polygon {

		public:
			// Constructors and destructors
			inline Polygon()                                     { alloc = false; n = 0; }
			inline Polygon(const int nn)                         { set_npoints(nn); }
			inline Polygon(const int nn, const Point &v)         { set_npoints(nn); set_points(v); c = compute_centroid(); }
			inline Polygon(const int nn, const Point *v)         { set_npoints(nn); set_points(v); c = compute_centroid(); }
			inline ~Polygon()                                    { clear(); }

			// Functions
			inline void   set_npoints(const int nn)              { n = nn; p = new Point[n+1]; alloc = true; }
			inline void   set_point(const int i, const Point &v) { if (alloc) p[i] = v; }
			inline void   set_points(const Point v)              { if (alloc) std::fill(p,p+n,v); }
			inline void   set_points(const Point *v)             { if (alloc) std::memcpy(p,v,n*sizeof(Point)); }
			inline void   set_centroid(const Point v)            { c = v; }
			inline void   set_bbox(Ball &b)                      { bbox = b; }
			inline void   clear()                                { n = 0; if (alloc) { delete [] p; } alloc = false; }
			inline void   set(const int nn, const Point &v)      { set_npoints(nn); set_points(v); }
			inline void   set(const int nn, const Point *v)      { set_npoints(nn); set_points(v); }
			inline Point *get_points() const                     { return p; }
			inline Point  get_point(const int i) const           { return p[i]; }
			inline Point  get_centroid() const                   { return c; }
			inline int    get_npoints() const                    { return n; }
			inline Ball   get_bbox() const                       { return bbox; }
			inline bool   isempty() const                        { return n == 0; }
			inline bool   isinside(const Point &v) const         { if (bbox > v) return ( (wn_PinPoly(this,v) != 0) ? true : false ); else return false; }
			inline bool   isinside_cn(const Point &v) const      { if (bbox > v) return ( (cn_PinPoly(this,v) == 1) ? true : false ); else return false; }
			inline bool   isinside_wn(const Point &v) const      { if (bbox > v) return ( (wn_PinPoly(this,v) != 0) ? true : false ); else return false; }
			
			inline void   print() const                          { for(int i=0; i<n; ++i) { printf("Point %d ",i); p[i].print(); printf("\n"); } }
			
			Point  compute_centroid();
			void   rotate(const double theta[3], const Point o);
			void   rotate_rodrigues(const double theta, const double k[3], const Point o);

			// Operators
			inline Point  operator[](int i) const                { return (i>=0) ? p[i] : p[n+i]; }
			inline Point &operator[](int i)                      { return (i>=0) ? p[i] : p[n+i]; }
			inline bool   operator==(const Polygon &pp) const;
			inline bool   operator!=(const Polygon &pp) const;
			inline bool   operator>(const Point &v) const;
			inline bool   operator<(const Point &v) const;

		private:
			bool   alloc;
			int    n;
			Point *p;
			Point  c;
			Ball   bbox;
	};

	// Point
	inline Point  &Point::operator=(const Point &pp)        { set(pp[0],pp[1],pp[2]); return (*this); }
	inline Point   Point::operator+(const Vector &vv) const { return Point(p[0]+vv[0],p[1]+vv[1],p[2]+vv[2]); }
	inline Point   Point::operator-(const Vector &vv) const { return Point(p[0]-vv[0],p[1]-vv[1],p[2]-vv[2]); }		
	inline Vector  Point::operator-(const Point &pp) const  { return Vector(p[0]-pp[0],p[1]-pp[1],p[2]-pp[2]); }
	inline bool    Point::operator==(const Point &v) const  { return ( (p[0] == v[0]) && (p[1] == v[1]) && (p[2] == v[2]) ); }
	inline bool    Point::operator!=(const Point &v) const  { return ( !((*this) == v )); }
	inline double  Point::dist(const Point &pp) const       { Vector d = (*this) - pp; return d.norm();  }
	inline double  Point::dist2(const Point &pp) const      { Vector d = (*this) - pp; return d.norm2(); }

	// Vector
	inline Vector &Vector::operator=(const Vector &vv)                { set(vv[0],vv[1],vv[2]); return (*this); }
	inline Vector  Vector::operator+(const Vector &vv) const          { return Vector(v[0]+vv[0],v[1]+vv[1],v[2]+vv[2]); }
	inline Vector  Vector::operator-(const Vector &vv) const          { return Vector(v[0]-vv[0],v[1]-vv[1],v[2]-vv[2]); }
	inline Vector  Vector::operator*(const double a) const            { return Vector(v[0]*a,v[1]*a,v[2]*a); }
	inline Vector  Vector::operator/(const double a) const            { return Vector(v[0]/a,v[1]/a,v[2]/a); }
	inline double  Vector::operator*(const Vector &vv) const          { return dot(vv); }
	inline Vector  Vector::operator^(const Vector &vv) const          { return cross(vv); }

	// Polygon
	inline bool Polygon::operator==(const Polygon &pp) const {
		// Check if polygons have the same number of points
		if (this->n != pp.get_npoints()) return false;
		// Loop points and check whether they are equal
		for (int ii = 0; ii < this->n; ++ii) {
			if (this->p[ii] != pp[ii]) return false;
		}
		return true;
	}
	inline bool Polygon::operator!=(const Polygon &pp) const { return ( !((*this)==pp) ); }
	inline bool Polygon::operator>(const Point &v) const     { return ( isinside(v) ); }   // Return true if Point inside Polygon
	inline bool Polygon::operator<(const Point &v) const     { return ( !isinside(v) ); }  // Return true if Point outside Polygon
}

#endif
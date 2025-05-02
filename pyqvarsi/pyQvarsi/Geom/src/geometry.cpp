/* 
	GEOMETRY
	
	Definition of a generic geometric tools such as polygons.

	Arnau Miro (OGS) (c) 2019
*/

#include <cmath>

#include "matrixMN.h"
#include "geometry.h"

namespace Geom
{
	/* FASTBALL

		Get a fast approximation for the 2D bounding ball 
		(based on the algorithm given by [Jack Ritter, 1990]).

		Input:  A polygon
		Output: Nothing, sets the ball class

		from: http://geomalgorithms.com/a08-_containers.html

		Copyright 2001 softSurfer, 2012 Dan Sunday
		This code may be freely used and modified for any purpose
		providing that this copyright notice is included with it.
		SoftSurfer makes no warranty for this code, and cannot be held
		liable for any real or imagined damage resulting from its use.
		Users of this code must verify correctness for their application.
	*/
	void Ball::fastBall(const Polygon &p) {

		double xmin,  xmax,  ymin,  ymax,  zmin,  zmax; // bounding box extremes
		int   Pxmin, Pxmax, Pymin, Pymax, Pzmin, Pzmax; // index of P[] at box extreme

		// Find a large diameter to start with
		// first get the bounding box and P[] extreme points for it
		xmin = xmax = p[0][0];
		ymin = ymax = p[0][1];
		zmin = zmax = p[0][2];
		Pxmin = Pxmax = Pymin = Pymax = Pzmin = Pzmax = 0;

		for (int ii=1; ii<p.get_npoints(); ++ii) {
			if (p[ii][0] < xmin) {
				xmin = p[ii][0]; Pxmin = ii;
			} else if (p[ii][0] > xmax) {
				xmax = p[ii][0]; Pxmax = ii;
			}
			if (p[ii][1] < ymin) {
				ymin = p[ii][1]; Pymin = ii;
			} else if (p[ii][1] > ymax) {
				ymax = p[ii][1]; Pymax = ii;
			}
			if (p[ii][2] < zmin) {
				zmin = p[ii][2]; Pzmin = ii;
			} else if (p[ii][2] > zmax) {
				zmax = p[ii][2]; Pzmax = ii;
			}
		}

		// Select the largest extent as an initial diameter for the  ball
		Point C;
		Vector dPx = p[Pxmax] - p[Pxmin], dPy = p[Pymax] - p[Pymin], dPz = p[Pzmax] - p[Pzmin];
		double rad2, dx2 = dPx.norm2(), dy2 = dPy.norm2(), dz2 = dPz.norm2();

		if (dx2 >= dy2 && dx2 >= dz2) {  // x direction is largest extent
			C    = p[Pxmin] + (dPx/2.); 
			rad2 = p[Pxmax].dist2(C);
		} else if (dy2 >= dx2 && dy2 >= dz2) { // y direction is largest extent
			C    = p[Pymin] + (dPy/2.);
			rad2 = p[Pymax].dist2(C);
		} else { // z direction is largest extent
			C    = p[Pzmin] + (dPz/2.);
			rad2 = p[Pzmax].dist2(C);
		}

		double rad = std::sqrt(rad2);

		// Now check that all points p[i] are in the ball
		// and if not, expand the ball just enough to include them
    	Vector dP; double dist, dist2;

    	for (int ii=0; ii<p.get_npoints(); ++ii) {
    		dP    = p[ii] - C; 
    		dist2 = dP.norm2();
    		if (dist2 <= rad2) continue; // p[i] is inside the ball already
    		// p[i] not in ball, so expand ball  to include it
    		// enlarge radius just enough
    		dist = std::sqrt(dist2); 
    		rad  = 0.5*(rad + dist); 
    		rad2 = rad*rad; 
    		C    = C + dP*((dist-rad)/dist); // shift Center toward p[i]
    	}

    	// Set ball parameters
    	this->set_center(C);
    	this->set_radius(rad);
	}

	/* CN_PINPOLY

		Crossing number test for a point in a polygon.

		Input:   P = a point,
		Return:  0 = outside, 1 = inside
		
		This code is patterned after [Franklin, 2000]
		from: http://geomalgorithms.com/a03-_inclusion.html

		Copyright 2001 softSurfer, 2012 Dan Sunday
		This code may be freely used and modified for any purpose
		providing that this copyright notice is included with it.
		SoftSurfer makes no warranty for this code, and cannot be held
		liable for any real or imagined damage resulting from its use.
		Users of this code must verify correctness for their application.
	*/
	int cn_PinPoly(const Polygon *poly, const Point &P) {

		int cn = 0; // The crossing number counter

		// Loop through all edges of the Polygon
		for (int ii=0; ii<poly->get_npoints(); ++ii) {
			if ( (((*poly)[ii][1] <= P[1]) && ((*poly)[ii+1][1] >  P[1]))         // an upward crossing
			  || (((*poly)[ii][1] >  P[1]) && ((*poly)[ii+1][1] <= P[1])) ) {     // a downward crossing

			  	// Compute  the actual edge-ray intersect x-coordinate
				double vt = (double)( (P[1]  - (*poly)[ii][1]) / ((*poly)[ii+1][1] - (*poly)[ii][1]) );
				if (P[0] <  (*poly)[ii][0] + vt * ((*poly)[ii+1][0] - (*poly)[ii][0]))  // P.x < intersect
					++cn; // A valid crossing of y=P.y right of P.x		
			}
		}
		return(cn & 1); // 0 if even (out), and 1 if  odd (in)
	}

	/* WN_PINPOLY

		Winding number test for a point in a polygon.

		Input:   P = a point,
		Return:  wn = the winding number (=0 only when P is outside)

		from: http://geomalgorithms.com/a03-_inclusion.html

		Copyright 2001 softSurfer, 2012 Dan Sunday
		This code may be freely used and modified for any purpose
		providing that this copyright notice is included with it.
		SoftSurfer makes no warranty for this code, and cannot be held
		liable for any real or imagined damage resulting from its use.
		Users of this code must verify correctness for their application.
	*/
	int wn_PinPoly(const Polygon *poly, const Point &P) {

		int wn = 0; // The  winding number counter

		// Loop through all edges of the polygon
		for (int ii=0; ii<poly->get_npoints(); ++ii) {   // edge from V[i] to  V[i+1]
			if ((*poly)[ii][1] <= P[1]) {   	// start y <= P.y
				if ((*poly)[ii+1][1] > P[1])			// an upward crossing
					if ( P.isLeft((*poly)[ii],(*poly)[ii+1]) > 0 ) // P left of  edge
						++wn; // have  a valid up intersect
			} else {                        // start y > P.y (no test needed)
				if ((*poly)[ii+1][1] <= P[1])	// a downward crossing
					if ( P.isLeft((*poly)[ii],(*poly)[ii+1]) < 0 ) // P left of  edge
						--wn; // have  a valid down intersect
			}
		}
		return wn;
	}

	/* COMPUTE_CENTROID

		Returns the centroid (Point) of a (2D) polygon.	
		3D version to be implemented.

		https://wwwf.imperial.ac.uk/~rn/centroid.pdf
		https://en.wikipedia.org/wiki/Centroid
	*/
	Point Polygon::compute_centroid() {
		double Cx = 0., Cy = 0., A = 0.;
		for (int ip=0; ip<this->get_npoints(); ++ip) {
			Cx += (this->get_point(ip)[0]   + this->get_point(ip+1)[0])*
				  (this->get_point(ip)[0]   * this->get_point(ip+1)[1] -
			       this->get_point(ip+1)[0] * this->get_point(ip)[1]);
			Cy += (this->get_point(ip)[1]   + this->get_point(ip+1)[1])* 
			      (this->get_point(ip)[0]   * this->get_point(ip+1)[1] -
			       this->get_point(ip+1)[0] * this->get_point(ip)[1]);
			A  +=  this->get_point(ip)[0]   * this->get_point(ip+1)[1] -
			       this->get_point(ip+1)[0] * this->get_point(ip)[1];
		}
		return Point(Cx/(3*A),Cy/(3*A),0.);
	}

	/* ROTATE
		
		Rotate a polygon by a theta radians 3D angle array 
		wrt to an origin Point (o).
	*/
	void Polygon::rotate(const double theta[3], const Point o) {
		// Compute sin and cos
		double cx = cos(theta[0]), sx = sin(theta[0]);
		double cy = cos(theta[1]), sy = sin(theta[1]);
		double cz = cos(theta[2]), sz = sin(theta[2]);
		// Build rotation matrices
		double valx[] = {1.,0.,0.,0.,cx,-sx,0.,sx,cx};
		double valy[] = {cy,0.,sy,0.,1.,0.,-sy,0.,cy};
		double valz[] = {cz,-sz,0.,sz,cz,0.,0.,0.,1.};
		matMN::matrixMN<double> Rx(3,3,valx), Ry(3,3,valy), Rz(3,3,valz);
		// Compute rotation matrix R
		matMN::matrixMN<double> R = Rx^Ry^Rz;
		// Project the points
		Point p;
		matMN::matrixMN<double> out(3,1), points(3,1);
		for (int ip=0; ip<this->get_npoints()+1; ++ip) {
			// Set the points matrix
			points[0][0] = this->get_point(ip)[0] - o[0];
			points[1][0] = this->get_point(ip)[1] - o[1];
			points[2][0] = this->get_point(ip)[2] - o[2];
			// Compute the projection
			out = R^points;
			// Set the output point
			p[0] = out[0][0] + o[0];
			p[1] = out[1][0] + o[1];
			p[2] = out[2][0] + o[2];
			// Set the point on the polygon
			this->set_point(ip,p);
		}
	}

	/* ROTATE_RODRIGUES

		Rotate a polygon by a theta radians around a 3-D unit vector axis (k),
		wrt to an origin Point (o).
		This implements the general matrix rotation formula by Rodrigues:
		https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
	*/
	void Polygon::rotate_rodrigues(const double theta, const double k[3], const Point o) {
		// Compute Rodrigues rotation matrix
		double valI[] = {1.,0.,0.,0.,1.,0.,0.,0.,1.};
		double valK[] = {0., -k[2], k[1], k[2], 0., -k[0], -k[1], k[0], 0.};
		matMN::matrixMN<double> I(3,3,valI), K(3,3,valK);
		matMN::matrixMN<double> R = I + sin(theta)*K + (1.-cos(theta))*(K^K);
		// Project the points
		Point p;
		matMN::matrixMN<double> out(3,1), points(3,1);
		for (int ip=0; ip<this->get_npoints()+1; ++ip) {
			// Set the points matrix
			points[0][0] = this->get_point(ip)[0] - o[0];
			points[1][0] = this->get_point(ip)[1] - o[1];
			points[2][0] = this->get_point(ip)[2] - o[2];
			// Compute the projection
			out = R^points;
			// Set the output point
			p[0] = out[0][0] + o[0];
			p[1] = out[1][0] + o[1];
			p[2] = out[2][0] + o[2];
			// Set the point on the polygon
			this->set_point(ip,p);
		}
	}
}
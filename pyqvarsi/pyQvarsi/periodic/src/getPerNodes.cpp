#include <iostream>
#include <fstream>
#include <map>
#include <set>
#include <math.h>
#include "assert.h"
#include <stdlib.h>
#include "D3class.h"

using namespace std;
double gzero2 = 1e-10;

int main(int argc,char **argv) {

    ofstream myfile,myfile2;
    ifstream infile,sfile;
	string name(argv[1]);
    
	double value1 = 0.0;
    double value2 = 0.0;
	int perDim    = 0;
	string perName= "x";

	if( argc > 3 )
	{
		value1 = atof(argv[2]); 
		value2 = atof(argv[3]); 
	}

	if( argc > 4 )
	{
		perDim =atol(argv[4]);
	}

	int dimensions = 2;
	if( argc > 5 )
	{
		dimensions =atol(argv[5]);
	}


	if( perDim == 0 )
	{
		perName = "x";
	}else if( perDim == 1 )
	{
		perName = "y";
	}else if( perDim == 2 )
	{
		perName = "z";
	}

    myfile.open (name+"."+perName+".per");
    infile.open (name+".coord");


	
	bool dim2DCase = true;
	if(dimensions==3)
	{
		dim2DCase = false;
	}

    int id;
    double x,y,z,goal;
    map<D3,int> no1, no2;

    if(infile.is_open()){
	    cout<<" parsing COORDINATES "<<endl;
	    while(!infile.eof()) {
			id   = -66666; 
			goal = 1.0e24;
			x    = 1.0e24;
			y    = 1.0e24;
			z    = 1.0e24;
			if( dim2DCase ) {
				infile>>id>>x>>y;
				z=0.0;
			}else{
		    	infile>>id>>x>>y>>z;
			}
	
			
			if( perDim == 0 )
			{
				goal = x;
				x = 0.0;
			}else if( perDim == 1 )
			{
				goal = y;
				y = 0.0;
			}else if( perDim == 2 )
			{
				goal = z;
				z = 0.0;
			}


		    if( fabs( goal -value1)<gzero2 ) {
			    no1 [D3( x,y,z )] = id; 
		    }
		    if( fabs( goal -value2)<gzero2 ) {
			    no2[D3( x,y,z )] = id; 
		    }
	    } 
    }

    cout<<" writting file begin"<<endl;

    for(map<D3,int>::iterator i=no1.begin(); i!=no1.end(); i++) {
       for( map<D3,int>::iterator k = no2.begin(); k!=no2.end(); k++){
          if (i->first == k->first){
              bool crit1( fabs(i->first[perDim] - value1) < gzero2);
              bool crit2( fabs(i->first[perDim] - value2) < gzero2);
              if (!crit1 || !crit2)
              {
                myfile << i->second << " "<< k->second << endl;
					 no2.erase(k);
					 break;
              }
          }
       }
    }

    cout<<" writting file end"<<endl;

    myfile.close();
    infile.close();

    return 0;
}

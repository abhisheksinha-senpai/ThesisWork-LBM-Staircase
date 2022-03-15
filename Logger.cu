#include "utilities.h"
#include "lbm.h"
#include <math.h>
#include <fstream>
using namespace std;

__host__ void logger(const char* name, float *gpu_src, unsigned int n)
{
    int ndigits = floor(log10((double)NSTEPS)+1.0);
    char filename[128];
    char format[32];
    sprintf(filename,"%s%d.csv",name,n);
    checkCudaErrors(cudaMemcpy(scalar_host,gpu_src,mem_size_scalar,cudaMemcpyDeviceToHost));
    
    ofstream o;
    o.open(filename);
    for(int i=0;i<mem_size_scalar;i++)
        o<<scalar_host[i]<<",";
    o<<endl;
    o.close();
}
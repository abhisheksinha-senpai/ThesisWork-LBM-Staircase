#include "utilities.h"
#include "lbm.h"
#include <iomanip>
#include <fstream>
#include "boundary.h"
using namespace std;

__host__ void logger(const char* name, float *gpu_src, unsigned int n)
{
    int ndigits = floor(log10((double)NSTEPS)+1.0);
    char filename[128];
    char format[32];
    sprintf(filename,"00_%s%d.csv",name,n);
    checkCudaErrors(cudaMemcpy(scalar_host,gpu_src,mem_size_scalar,cudaMemcpyDeviceToHost));
    
    ofstream o;
    o.open(filename);
    for(int i=0;i<NZ;i++)
    {
        for(int j=0;j<NY;j++)
        {
            //setprecision(5);
            for(int k = 0;k<NX;k++)
                o<<scalar_host[i*(NX*NY) + NX*j + k]<<","<<i<<","<<j<<","<<k<<","<<cpu_boundary[i*(NX*NY) + NX*j + k]<<endl;
        }
    }
    o<<endl;
    o.close();
}
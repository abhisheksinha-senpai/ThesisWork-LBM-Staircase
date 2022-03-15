#include "utilities.h"
#include "lbm.h"
#include <math.h>

__host__ void logger(const char* name, float *gpu_src, unsigned int n)
{
    int ndigits = floor(log10((double)NSTEPS)+1.0);
    char filename[128];
    char format[32];
    sprintf(filename,"%s%d.txt",name,n);
    checkCudaErrors(cudaMemcpy(scalar_host,gpu_src,mem_size_scalar,cudaMemcpyDeviceToHost));
    FILE *fout = fopen(filename,"w");
    
    // write data
    fwrite(scalar_host,1,mem_size_scalar,fout);
    
    // close file
    fclose(fout);
    
    if(ferror(fout))
    {
        fprintf(stderr,"\nError saving to %s\n",filename);
        perror("\n");
    }
    else
    {
        printf("\nSaved to %s\n",filename);
    }
}
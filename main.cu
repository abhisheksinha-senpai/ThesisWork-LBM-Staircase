#include "utilities.h"
#include "boundary.h"
#include "lbm.h"

int main(int argc, char* argv[])
{
    //cudaDeviceReset();
    getDeviceInfo();
    // cudaEvent_t start, stop;

    AllocateMemory();

    // checkCudaErrors(cudaEventCreate(&start));
    // checkCudaErrors(cudaEventCreate(&stop));

    cpu_field_Initialization();
    cpu_equi_Initialization();
    float *temp;
    bool save = false;
    for(unsigned n=0;n<NSTEPS;n++)
    {
        save = ((n+1)%NSAVE == 0);
        if(save)
        {
            logger("rho",rho_gpu,n+1);
            logger("ux", ux_gpu, n+1);
            logger("uy", uy_gpu, n+1);
            logger("uz", uz_gpu, n+1);
        }
        cpu_stream_collide(save);
        temp = f1_gpu;
        f1_gpu = f2_gpu;
        f2_gpu = temp;
    }
   DeallocateMemory();

    // checkCudaErrors(cudaEventDestroy(start));
    // checkCudaErrors(cudaEventDestroy(stop));
    return 0;
}
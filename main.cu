#include "utilities.h"
#include "boundary.h"
#include "lbm.h"

int main(int argc, char* argv[])
{
    cudaDeviceReset();
    getDeviceInfo();
    // cudaEvent_t start, stop;

    AllocateMemory();

    // checkCudaErrors(cudaEventCreate(&start));
    // checkCudaErrors(cudaEventCreate(&stop));

    cpu_field_Initialization();
    cpu_field_Initialization();

    for(usnsigned n=0;n<NSTEPS;n++)
    {
        
    }
    DeallocateMemory();

    // checkCudaErrors(cudaEventDestroy(start));
    // checkCudaErrors(cudaEventDestroy(stop));
    return 0;
}
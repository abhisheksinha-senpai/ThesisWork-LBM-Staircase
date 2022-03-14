#include "utilities.h"
#include "lbm.h"
#include "boundary.h"

using namespace std;

unsigned int mem_size_props;
float *f0_gpu,*f1_gpu,*f2_gpu;
float *rho_gpu,*ux_gpu,*uy_gpu, *uz_gpu;
float *prop_gpu;
float *scalar_host;

void getDeviceInfo()
{
    double bytesPerMiB = 1024.0*1024.0;
    double bytesPerGiB = 1024.0*1024.0*1024.0;
    
    checkCudaErrors(cudaSetDevice(0));
    int deviceId = 0;
    checkCudaErrors(cudaGetDevice(&deviceId));
    
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, deviceId));
    
    size_t gpu_free_mem, gpu_total_mem;
    checkCudaErrors(cudaMemGetInfo(&gpu_free_mem,&gpu_total_mem));

    printf("CUDA information\n");
    printf("       using device: %d\n", deviceId);
    printf("               name: %s\n",deviceProp.name);
    printf("    multiprocessors: %d\n",deviceProp.multiProcessorCount);
    printf(" compute capability: %d.%d\n",deviceProp.major,deviceProp.minor);
    printf("      global memory: %.1f MiB\n",deviceProp.totalGlobalMem/bytesPerMiB);
    printf("        free memory: %.1f MiB\n",gpu_free_mem/bytesPerMiB);
    return;
}

void AllocateMemory()
{
    double bytesPerMiB = 1024.0*1024.0;
    checkCudaErrors(cudaMalloc((void**)&f0_gpu,mem_size_0dir));
    checkCudaErrors(cudaMalloc((void**)&f1_gpu,mem_size_n0dir));
    checkCudaErrors(cudaMalloc((void**)&f2_gpu,mem_size_n0dir));
    checkCudaErrors(cudaMalloc((void**)&rho_gpu,mem_size_scalar));
    checkCudaErrors(cudaMalloc((void**)&ux_gpu,mem_size_scalar));
    checkCudaErrors(cudaMalloc((void**)&uy_gpu,mem_size_scalar));
    checkCudaErrors(cudaMalloc((void**)&uz_gpu,mem_size_scalar));
    checkCudaErrors(cudaMalloc((void**)&gpu_boundary,mem_size_bound));
    checkCudaErrors(cudaMalloc((void**)&gpu_normals,mem_size_normal));
    mem_size_props = 0;//7*NX/nThreads*NY*sizeof(float);
    // checkCudaErrors(cudaMalloc((void**)&prop_gpu,mem_size_props));

    scalar_host  = (float*) malloc(mem_size_scalar);
    cpu_boundary = (bool *)malloc(mem_size_bound);
    cpu_normals = (short *)malloc(mem_size_normals);
    if(scalar_host == NULL || cpu_boundary == NULL || cpu_normals == NULL )
    {
        fprintf(stderr,"Error: unable to allocate required host memory (%.1f MiB).\n",mem_size_scalar/bytesPerMiB);
        exit(-1);
    }
    else
    {
        unsigned int gpu_total = mem_size_0dir + 2*mem_size_n0dir + 4* mem_size_scalar + mem_size_props + mem_size_bound + mem_size_normals;
        printf("Allocated %.1f MiB memory in CPU and %.1f MiB in GPU\n",mem_size_scalar/bytesPerMiB, gpu_total/bytesPerMiB);
    }
}

void DeallocateMemory()
{
  
    // free all memory allocatd on the GPU and host
    checkCudaErrors(cudaFree(f0_gpu));
    checkCudaErrors(cudaFree(f1_gpu));
    checkCudaErrors(cudaFree(f2_gpu));
    checkCudaErrors(cudaFree(rho_gpu));
    checkCudaErrors(cudaFree(ux_gpu));
    checkCudaErrors(cudaFree(uy_gpu));
    checkCudaErrors(cudaFree(uz_gpu));
    // checkCudaErrors(cudaFree(prop_gpu));
    checkCudaErrors(cudaFree(gpu_boundary));
    checkCudaErrors(cudaFree(gpu_normals));
    free(scalar_host);
    free(cpu_boundary);
    free(cpu_normals);
    
    // release resources associated with the GPU device
    cudaDeviceReset();
}
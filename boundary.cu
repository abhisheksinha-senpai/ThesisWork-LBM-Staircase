#include "boundary.h"
#include "lbm.h"
#include "utilities.h"

using namespace std;

__host__ void cpu_field_Initialization()
{
    defineBoundary();
    checkCudaErrors(cudaMemcpy(gpu_boundary, cpu_boundary, mem_size_bound,  cudaMemcpyHostToDevice));
    // blocks in grid
    dim3  grid(NX/nThreads, NY, NZ);
    // threads in block
    dim3  threads(nThreads, 1, 1);

    gpu_field_Initialization<<< grid, threads >>>(gpu_boundary, rho_gpu, ux_gpu, uy_gpu, uz_gpu);
    getLastCudaError("gpu_field_Initialization kernel error");
}

__global__ void gpu_field_Initialization(bool *boundary, float *rho, float *ux, float *uy, float *uz)
{
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int tz = threadIdx.z;
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;
    unsigned int bz = blockIdx.y;
    unsigned int bw = blockDim.x;
    unsigned int bh = blockDim.y;
    unsigned int bd = blockDim.z;
    unsigned int idx = tx + bx * bw;
    unsigned int idy = ty + by * bh;
    unsigned int idz = tz + bz * bd;

    unsigned int sidx = gpu_scalar_index(idx, idy, idz);
    rho[sidx] = (boundary[sidx] == true)?1:0;
    ux[sidx] = 0.0;
    uy[sidx] = 0.0;
    uz[sidx] = 0.0;
}

__host__ void defineBoundary()
{
    for(int k=0;k<NZ;k++)
    {
        // cout<<"first loop"<<endl;
        for(int i=0;i<NY;i++)
        {
            // cout<<"sec loop"<<endl;
            for(int j=0;j<NX;j++)
            {
                // cout<<"third loop"<<endl;
                if(j<NX/8)
                {
                    if(i<NY/4 || i>NY/2)
                    {
                        // cout<<"1     "<<k*NX*NY + i*NX +j<<endl;
                        cpu_boundary[k*NX*NY + i*NX +j] = true;
                    }
                    else
                    {
                        // cout<<"2     "<<k*NX*NY + i*NX +j<<endl;
                        cpu_boundary[k*NX*NY + i*NX +j] = false;
                    }
                }
                else if(j<NX/4)
                {
                    if(i>(NY-NY/4) || i<NY/4)
                    {
                        // cout<<"3     "<<k*NX*NY + i*NX +j<<endl;
                        cpu_boundary[k*NX*NY + i*NX +j] = true;
                    }
                    else
                    {
                        // cout<<"4     "<<k*NX*NY + i*NX +j<<endl;
                        cpu_boundary[k*NX*NY + i*NX +j] = false;
                    }
                }
                else if(j<(NX/2 - NX/4))
                {
                    if(i<NY/2)
                    {
                        // cout<<"5     "<<k*NX*NY + i*NX +j<<endl;
                        cpu_boundary[k*NX*NY + i*NX +j] = true;
                    }
                    else
                    {
                        // cout<<"6     "<<k*NX*NY + i*NX +j<<endl;
                        cpu_boundary[k*NX*NY + i*NX +j] = false;
                    }
                }
                else  if(j<(NX/2))
                {
                    if(i<(NY - NY/4))
                    {
                        // cout<<"7     "<<k*NX*NY + i*NX +j<<endl;
                        cpu_boundary[k*NX*NY + i*NX +j] = true;
                    }
                    else
                    {
                        // cout<<"8     "<<k*NX*NY + i*NX +j<<endl;
                        cpu_boundary[k*NX*NY + i*NX +j] = false;
                    }
                }
                else
                {
                    // cout<<"9     "<<k*NX*NY + i*NX +j<<endl;
                    cpu_boundary[k*NX*NY + i*NX +j] = false;
                }            
            }
        }
    }
}
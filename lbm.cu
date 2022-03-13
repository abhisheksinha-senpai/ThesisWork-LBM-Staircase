#include "lbm.h"

__global__ void gpu_equi_Initialization(bool *boundary, float *rho, float *ux, float *uy, float *uz)
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

    unsigned int sidx = gpu_scalar_index(idx,idy,idz);
    float lattice_rho = rho[sidx];
    float lattice_ux  = ux[sidx];
    float lattice_uy  = uy[sidx];
    float lattice_uz  = uz[sidx];
    float w0r = w0*lattice_rho;
    float wsr = ws*lattice_rho;
    float wdr = wd*lattice_rho;

    float omusq = 1.0 - 1.5*(lattice_ux*lattice_ux+lattice_uy*lattice_uy+lattice_uz*lattice_uz);
    float tux = 3.0*lattice_ux;
    float tuy = 3.0*lattice_uy;
    float tux = 3.0*lattice_uz;
}

__host__ void cpu_equi_Initialization()
{
    // blocks in grid
    dim3  grid(NX/nThreads, NY, NZ);
    // threads in block
    dim3  threads(nThreads, 1, 1);
    gpu_equi_Initialization<<< grid, threads >>>(gpu_boundary, rho_gpu, ux_gpu, uy_gpu, uz_gpu);
    getLastCudaError("gpu_equi_Initialization kernel error");
}
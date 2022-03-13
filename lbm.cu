#include "lbm.h"

__device__ void apply(float *field,int indexX, int indexY, int indexZ,
     float omusq, float tux, float tuy, float tuz, float wsr, float wdr)
{
    float cidot3u = tux;
    field[gpu_fieldn_index(x,y,z,1)]  = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = tuy;
    field[gpu_fieldn_index(x,y,z,2)]  = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = tuz;
    field[gpu_fieldn_index(x,y,z,3)]  = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));

    cidot3u = -tux;
    field[gpu_fieldn_index(x,y,z,4)]  = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = -tuy;
    field[gpu_fieldn_index(x,y,z,5)]  = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = -tuz;
    field[gpu_fieldn_index(x,y,z,6)]  = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));

    cidot3u = tux+tuy;
    field[gpu_fieldn_index(x,y,z,7)]  = wsd*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = tuz+tuy;
    field[gpu_fieldn_index(x,y,z,8)]  = wsd*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = tux+tuz;
    field[gpu_fieldn_index(x,y,z,9)]  = wsd*(omusq + cidot3u*(1.0+0.5*cidot3u));
    
    cidot3u = -tux-tuy;
    field[gpu_fieldn_index(x,y,z,10)]  = wsd*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = -tuz-tuy;
    field[gpu_fieldn_index(x,y,z,11)]  = wsd*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = -tux-tuz;
    field[gpu_fieldn_index(x,y,z,12)]  = wsd*(omusq + cidot3u*(1.0+0.5*cidot3u));

    cidot3u = -tux+tuy;
    field[gpu_fieldn_index(x,y,z,13)]  = wsd*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = tux-tuy;
    field[gpu_fieldn_index(x,y,z,14)]  = wsd*(omusq + cidot3u*(1.0+0.5*cidot3u));
    
    cidot3u = -tuz+tuy;
    field[gpu_fieldn_index(x,y,z,15)]  = wsd*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = tuz-tuy;
    field[gpu_fieldn_index(x,y,z,16)]  = wsd*(omusq + cidot3u*(1.0+0.5*cidot3u));
    
    cidot3u = -tux+tuz;
    field[gpu_fieldn_index(x,y,z,17)]  = wsd*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = +tux-tuz;
    field[gpu_fieldn_index(x,y,z,18)]  = wsd*(omusq + cidot3u*(1.0+0.5*cidot3u));
}

__global__ void gpu_equi_Initialization(float *f0, float *f1, float *rho, float *ux, float *uy, float *uz)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.y * blockDim.z;

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
    float tuz = 3.0*lattice_uz;

    f0[gpu_field0_index(x,y,z)] = w0r*(omusq);
    
    apply(f1,idx, idy, idz, omusq, tux, tuy, tuz, wsr, wsd);

}

__host__ void cpu_equi_Initialization()
{
    // blocks in grid
    dim3  grid(NX/nThreads, NY, NZ);
    // threads in block
    dim3  threads(nThreads, 1, 1);
    gpu_equi_Initialization<<< grid, threads >>>(f0_gpu, f1_gpu, rho_gpu, ux_gpu, uy_gpu, uz_gpu);
    getLastCudaError("gpu_equi_Initialization kernel error");
}

__host__ void cpu_stream_collide()
{
    dim3  grid(NX/nThreads, NY, 1);
    // threads in block
    dim3  threads(nThreads, 1, 1);

    gpu_stream_collide<<< grid, threads >>>(f0_gpu, f1_gpu, f2_gpu, rho_gpu,
        ux_gpu, uy_gpu, uz_gpu);
    getLastCudaError("gpu_stream_collide kernel error");
}

__global__ void gpu_stream_collide(float *f0, float *f1, float *f2, float *rho
, float *ux, float *uy, float *uz)
{
    const double tauinv = 2.0/(6.0*nu+1.0); // 1/tau
    const double omtauinv = 1.0-tauinv;
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.y * blockDim.z;
}
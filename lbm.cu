#include "lbm.h"

__device__ void collide(float *field,int indexX, int indexY, int indexZ,
     float omusq, float tux, float tuy, float tuz, float wsr, float wdr)
{
    float cidot3u = tux;
    field[gpu_fieldn_index(x,y,z,1)]  = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = -tux;
    field[gpu_fieldn_index(x,y,z,2)]  = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
   
    cidot3u = tuy;
    field[gpu_fieldn_index(x,y,z,3)]  = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = -tuy;
    field[gpu_fieldn_index(x,y,z,4)]  = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
    
    cidot3u = tuz;
    field[gpu_fieldn_index(x,y,z,5)]  = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = -tuz;
    field[gpu_fieldn_index(x,y,z,6)]  = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));

    cidot3u = tux+tuy;
    field[gpu_fieldn_index(x,y,z,7)]  = wsd*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = -tux-tuy;
    field[gpu_fieldn_index(x,y,z,8)]  = wsd*(omusq + cidot3u*(1.0+0.5*cidot3u));
    
    cidot3u = tux+tuz;
    field[gpu_fieldn_index(x,y,z,9)]  = wsd*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = -tux-tuz;
    field[gpu_fieldn_index(x,y,z,10)]  = wsd*(omusq + cidot3u*(1.0+0.5*cidot3u));
    
    cidot3u = tuz+tuy;
    field[gpu_fieldn_index(x,y,z,11)]  = wsd*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u =-tuz-tuy;
    field[gpu_fieldn_index(x,y,z,12)]  = wsd*(omusq + cidot3u*(1.0+0.5*cidot3u));

    cidot3u = tux-tuy;
    field[gpu_fieldn_index(x,y,z,13)]  = wsd*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = -tux+tuy;
    field[gpu_fieldn_index(x,y,z,14)]  = wsd*(omusq + cidot3u*(1.0+0.5*cidot3u));
    
    cidot3u = +tux-tuz;
    field[gpu_fieldn_index(x,y,z,15)]  = wsd*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = -tux+tuz;
    field[gpu_fieldn_index(x,y,z,16)]  = wsd*(omusq + cidot3u*(1.0+0.5*cidot3u));
    
    cidot3u = tuy-tuz;
    field[gpu_fieldn_index(x,y,z,17)]  = wsd*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = -tuy+tuz;
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
    
    collide(f1,idx, idy, idz, omusq, tux, tuy, tuz, wsr, wsd);

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

    gpu_stream_collide<<< grid, threads >>>(gpu_boundary, f0_gpu, f1_gpu, f2_gpu, rho_gpu,
        ux_gpu, uy_gpu, uz_gpu);
    getLastCudaError("gpu_stream_collide kernel error");
}

__global__ void gpu_stream_collide(int4* boundary, float *f0, float *f1, float *f2, float *rho
, float *ux, float *uy, float *uz)
{
    const float tauinv = 2.0/(6.0*nu+1.0); // 1/tau
    const float omtauinv = 1.0-tauinv;
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.y * blockDim.z;

    int4 bound = boundary[gpu_scalar_index(idx, idy, idz)];

    unsigned int xp1 = (idx+1)%NX; //front in X direction
    unsigned int yp1 = (idy+1)%NY; //front in Y direction
    unsigned int zp1 = (idz+1)%NZ; //front in Z direction
    unsigned int xm1 = (NX+idx-1)%NX; //back in X direction
    unsigned int ym1 = (NY+idy-1)%NY; //back in Y direction
    unsigned int zm1 = (NZ+idz-1)%NZ; //back in Z direction

    enforce_boundary(f1, boundary, idx, idy, idz);

    float ft0 = f0[gpu_field0_index(idx,idy,idz)];

    // load populations from adjacent nodes
    float ft1 = f1[gpu_fieldn_index(xm1, idy, idz, 1)];
    float ft2 = f1[gpu_fieldn_index(xp1, idy, idz, 2)];
    float ft3 = f1[gpu_fieldn_index(idx, ym1, idz, 3)];
    float ft4 = f1[gpu_fieldn_index(idx, yp1, idz, 4)];
    float ft5 = f1[gpu_fieldn_index(idx, idy, zm1, 5)];
    float ft6 = f1[gpu_fieldn_index(idx, idy, zp1, 6)];
    float ft7 = f1[gpu_fieldn_index(xm1, ym1, idz, 7)];
    float ft8 = f1[gpu_fieldn_index(xp1, yp1, idz, 8)];
    float ft9 = f1[gpu_fieldn_index(xm1, idy, zm1, 9)];
    float ft10 = f1[gpu_fieldn_index(xp1, idy, zp1, 10)];
    float ft11 = f1[gpu_fieldn_index(idx, ym1, zm1, 11)];
    float ft12 = f1[gpu_fieldn_index(idx, yp1, zp1, 12)];
    float ft13 = f1[gpu_fieldn_index(xm1, yp1, idz, 13)];
    float ft14 = f1[gpu_fieldn_index(xp1, ym1, idz, 14)];
    float ft15 = f1[gpu_fieldn_index(xm1, idy, zp1, 15)];
    float ft16 = f1[gpu_fieldn_index(xp1, idy, zm1, 16)];
    float ft17 = f1[gpu_fieldn_index(idx, ym1, zp1, 17)];
    float ft18 = f1[gpu_fieldn_index(idx, yp1, zm1, 18)];

    float rho = (bound== 2)?1:ft0+ft1+ft2+ft3+ft4+ft5+ft6+ft7+ft8;
    float rhoinv = 1.0/rho;

    float ux = (bound== 2)?1:(bound==4)?0:rhoinv*(ft1+ft7+ft9+f13+f15-(ft2+ft8+ft10+f14+f16));
    float uy = (bound== 2)?0:(bound==4)?0:rhoinv*(ft3+ft7+ft11+f14+f17-(ft4+ft8+ft12+f13+f18));
    float uz = (bound== 2)?0:(bound==4)?0:rhoinv*(ft5+ft9+ft11+f16+f18-(ft6+ft10+ft12+f15+f17));

    float tw0r = tauinv*w0*rho; //   w[0]*rho/tau 
    float twsr = tauinv*ws*rho; // w[1-4]*rho/tau
    float twdr = tauinv*wd*rho; // w[5-8]*rho/tau
    float omusq = 1.0 - 1.5*(ux*ux+uy*uy+*uz*uz); // 1-(3/2)u.u

    float cidot3u = tux;
    float tux = 3.0*ux;
    float tuy = 3.0*uy;
    float tuz = 3.0*uz;

    f0[gpu_field0_index(x,y,z)] = w0r*(omusq);
    
    collide(f2,idx, idy, idz, omusq, tux, tuy, tuz, wsr, wsd);

}
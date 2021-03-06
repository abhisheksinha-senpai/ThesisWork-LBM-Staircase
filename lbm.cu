#include "lbm.h"
#include "utilities.h"
#include "boundary.h"

__device__ void collide(float *field,int x, int y, int z,
     float omusq, float tux, float tuy, float tuz, float wsr, float wsd)
{
    float cidot3u = tux;
    field[gpu_fieldn_index(x,y,z,1)]  =wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = -tux;
    field[gpu_fieldn_index(x,y,z,2)]  =wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
   
    cidot3u = tuy;
    field[gpu_fieldn_index(x,y,z,3)]  =wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = -tuy;
    field[gpu_fieldn_index(x,y,z,4)]  =wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
    
    cidot3u = tuz;
    field[gpu_fieldn_index(x,y,z,5)]  =wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = -tuz;
    field[gpu_fieldn_index(x,y,z,6)]  =wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));

    cidot3u = tux+tuy;
    field[gpu_fieldn_index(x,y,z,7)]  = wsd*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = -tux-tuy;
    field[gpu_fieldn_index(x,y,z,8)]  = wsd*(omusq + cidot3u*(1.0+0.5*cidot3u));
    
    cidot3u = tux+tuz;
    field[gpu_fieldn_index(x,y,z,9)]  = wsd*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = -tux-tuz;
    field[gpu_fieldn_index(x,y,z,10)]  =wsd*(omusq + cidot3u*(1.0+0.5*cidot3u));
    
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
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;

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

    f0[gpu_field0_index(idx,idy,idz)] = w0r*(omusq);
    
    collide(f1,idx, idy, idz, omusq, tux, tuy, tuz, wsr, wdr);

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

__host__ void cpu_stream_collide(bool save)
{
    dim3  grid(NX/nThreads, NY, NZ);
    // threads in block
    dim3  threads(nThreads, 1, 1);

    gpu_stream_collide<<< grid, threads >>>(gpu_boundary, f0_gpu, f1_gpu, f2_gpu, rho_gpu,
        ux_gpu, uy_gpu, uz_gpu, save);
    getLastCudaError("gpu_stream_collide kernel error");
}

__global__ void gpu_stream_collide(short* boundary, float *f0, float *f1, float *f2, float *rho
, float *ux, float *uy, float *uz, bool save)
{
    const float tauinv = 1/tau; // 1/tau
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.z * blockDim.z;

    short bound = boundary[gpu_scalar_index(idx, idy, idz)];

    unsigned int xp1 = (idx+1)%NX; //front in X direction
    unsigned int yp1 = (idy+1)%NY; //front in Y direction
    unsigned int zp1 = (idz+1)%NZ; //front in Z direction
    unsigned int xm1 = (NX+idx-1)%NX; //back in X direction
    unsigned int ym1 = (NY+idy-1)%NY; //back in Y direction
    unsigned int zm1 = (NZ+idz-1)%NZ; //back in Z direction

//    if(bound == 2 || bound == 5 || bound ==6 || bound == 7)
    // stream_enforce_boundary(f1, f0, boundary, idx, idy, idz);

    float ft0 = f0[gpu_field0_index(idx,idy,idz)];

    // load populations from adjacent nodes
    float ft1, ft2, ft3, ft4, ft5, ft6, ft7, ft8, ft9;
    float ft10, ft11, ft12, ft13, ft14, ft15, ft16, ft17, ft18;
    float lat_rho, rhoinv, lat_ux, lat_uy, lat_uz;
    if(bound == 1)
    {
        ft1 = f1[gpu_fieldn_index(xm1, idy, idz, 1)];
        ft2 = f1[gpu_fieldn_index(xp1, idy, idz, 2)];
        ft3 = f1[gpu_fieldn_index(idx, ym1, idz, 3)];
        ft4 = f1[gpu_fieldn_index(idx, yp1, idz, 4)];
        ft5 = f1[gpu_fieldn_index(idx, idy, zm1, 5)];
        ft6 = f1[gpu_fieldn_index(idx, idy, zp1, 6)];
        ft7 = f1[gpu_fieldn_index(xm1, ym1, idz, 7)];
        ft8 = f1[gpu_fieldn_index(xp1, yp1, idz, 8)];
        ft9 = f1[gpu_fieldn_index(xm1, idy, zm1, 9)];
        ft10 = f1[gpu_fieldn_index(xp1, idy, zp1, 10)];
        ft11 = f1[gpu_fieldn_index(idx, ym1, zm1, 11)];
        ft12 = f1[gpu_fieldn_index(idx, yp1, zp1, 12)];
        ft13 = f1[gpu_fieldn_index(xm1, yp1, idz, 13)];
        ft14 = f1[gpu_fieldn_index(xp1, ym1, idz, 14)];
        ft15 = f1[gpu_fieldn_index(xm1, idy, zp1, 15)];
        ft16 = f1[gpu_fieldn_index(xp1, idy, zm1, 16)];
        ft17 = f1[gpu_fieldn_index(idx, ym1, zp1, 17)];
        ft18 = f1[gpu_fieldn_index(idx, yp1, zm1, 18)];
    }
    else
    {
        ft1 = (boundary[gpu_scalar_index(xm1, idy, idz)] == 4)?f1[gpu_fieldn_index(idx, idy, idz, 2)]:f1[gpu_fieldn_index(xm1, idy, idz, 1)];
        ft2 = (boundary[gpu_scalar_index(xp1, idy, idz)] == 4)?f1[gpu_fieldn_index(idx, idy, idz, 1)]:f1[gpu_fieldn_index(xp1, idy, idz, 2)];
        ft3 = (boundary[gpu_scalar_index(idx, ym1, idz)] == 4)?f1[gpu_fieldn_index(idx, idy, idz, 4)]:f1[gpu_fieldn_index(idx, ym1, idz, 3)];
        ft4 = (boundary[gpu_scalar_index(idx, yp1, idz)] == 4)?f1[gpu_fieldn_index(idx, idy, idz, 3)]:f1[gpu_fieldn_index(idx, yp1, idz, 4)];
        ft5 = (boundary[gpu_scalar_index(idx, idy, zm1)] == 4)?f1[gpu_fieldn_index(idx, idy, idz, 6)]:f1[gpu_fieldn_index(idx, idy, zm1, 5)];
        ft6 = (boundary[gpu_scalar_index(idx, idy, zp1)] == 4)?f1[gpu_fieldn_index(idx, idy, idz, 5)]:f1[gpu_fieldn_index(idx, idy, zp1, 6)];
        ft7 = (boundary[gpu_scalar_index(xm1, ym1, idz)] == 4)?f1[gpu_fieldn_index(idx, idy, idz, 8)]:f1[gpu_fieldn_index(xm1, ym1, idz, 7)];
        ft8 = (boundary[gpu_scalar_index(xp1, yp1, idz)] == 4)?f1[gpu_fieldn_index(idx, idy, idz, 7)]:f1[gpu_fieldn_index(xp1, yp1, idz, 8)];
        ft9 = (boundary[gpu_scalar_index(xm1, idy, zm1)] == 4)?f1[gpu_fieldn_index(idx, idy, idz, 10)]:f1[gpu_fieldn_index(xm1, idy, zm1, 9)];
        ft10 = (boundary[gpu_scalar_index(xp1, idy, zp1)] == 4)?f1[gpu_fieldn_index(idx, idy, idz, 9)]:f1[gpu_fieldn_index(xp1, idy, zp1, 10)];
        ft11 = (boundary[gpu_scalar_index(idx, ym1, zm1)] == 4)?f1[gpu_fieldn_index(idx, idy, idz, 12)]:f1[gpu_fieldn_index(idx, ym1, zm1, 11)];
        ft12 = (boundary[gpu_scalar_index(idx, yp1, zp1)] == 4)?f1[gpu_fieldn_index(idx, idy, idz, 11)]:f1[gpu_fieldn_index(idx, yp1, zp1, 12)];
        ft13 = (boundary[gpu_scalar_index(xm1, yp1, idz)] == 4)?f1[gpu_fieldn_index(idx, idy, idz, 14)]:f1[gpu_fieldn_index(xm1, yp1, idz, 13)];
        ft14 = (boundary[gpu_scalar_index(xp1, ym1, idz)] == 4)?f1[gpu_fieldn_index(idx, idy, idz, 13)]:f1[gpu_fieldn_index(xp1, ym1, idz, 14)];
        ft15 = (boundary[gpu_scalar_index(xm1, idy, zp1)] == 4)?f1[gpu_fieldn_index(idx, idy, idz, 16)]:f1[gpu_fieldn_index(xm1, idy, zp1, 15)];
        ft16 = (boundary[gpu_scalar_index(xp1, idy, zm1)] == 4)?f1[gpu_fieldn_index(idx, idy, idz, 15)]:f1[gpu_fieldn_index(xp1, idy, zm1, 16)];
        ft17 = (boundary[gpu_scalar_index(idx, ym1, zp1)] == 4)?f1[gpu_fieldn_index(idx, idy, idz, 18)]:f1[gpu_fieldn_index(idx, ym1, zp1, 17)];
        ft18 = (boundary[gpu_scalar_index(idx, yp1, zm1)] == 4)?f1[gpu_fieldn_index(idx, idy, idz, 17)]:f1[gpu_fieldn_index(idx, yp1, zm1, 18)];

        if (bound == 2)
        {
            ft1 = f1[gpu_fieldn_index(idx, idy, idz, 2)] + 0.3*ws;
            ft9 = f1[gpu_fieldn_index(idx, idy, idz, 10)] + 0.3*wd;
            ft15 = f1[gpu_fieldn_index(idx, idy, idz, 16)] + 0.3*wd;
            ft13 = f1[gpu_fieldn_index(idx, idy, idz, 14)] + 0.3*wd;
            ft7 = f1[gpu_fieldn_index(idx, idy, idz, 8)] + 0.3*wd;
        }

        if(bound == 7)
        {
            
            // float on1 = f1[gpu_fieldn_index(idx, idy, idz,1)];
            // float on2 = f1[gpu_fieldn_index(idx, idy, idz,2)];
            // float on3 = f1[gpu_fieldn_index(idx, idy, idz,3)];
            // float on4 = f1[gpu_fieldn_index(idx, idy, idz,4)];
            // float on5 = f1[gpu_fieldn_index(idx, idy, idz,5)];
            // float on6 = f1[gpu_fieldn_index(idx, idy, idz,6)];
            // float on7 = f1[gpu_fieldn_index(idx, idy, idz,7)];
            // float on8 = f1[gpu_fieldn_index(idx, idy, idz,8)];
            // float on9 = f1[gpu_fieldn_index(idx, idy, idz,9)];
            // float on10 = f1[gpu_fieldn_index(idx, idy, idz,10)];
            // float on11 = f1[gpu_fieldn_index(idx, idy, idz,11)];
            // float on12 = f1[gpu_fieldn_index(idx, idy, idz,12)];
            // float on13 = f1[gpu_fieldn_index(idx, idy, idz,13)];
            // float on14 = f1[gpu_fieldn_index(idx, idy, idz,14)];
            // float on15 = f1[gpu_fieldn_index(idx, idy, idz,15)];
            // float on16 = f1[gpu_fieldn_index(idx, idy, idz,16)];
            // float on17 = f1[gpu_fieldn_index(idx, idy, idz,17)];
            // float on18 = f1[gpu_fieldn_index(idx, idy, idz,18)];

            // float lat_rho = ft0+on1+on2+on3+on4+on5+on6+on7+on8+on9+on10+on11+on12+on13+on14+on15+on16+on17+on18;
            // float rhoinv = min(100.0, 1.0/(lat_rho));

            // float lat_ux = rhoinv*(on1+on7+on9+on13+on15-(on2+on8+on10+on14+on16));
            // float lat_uy = rhoinv*(on3+on7+on11+on14+on17-(on4+on8+on12+on13+on18));
            // float lat_uz = rhoinv*(on5+on9+on11+on16+on18-(on6+on10+on12+on15+on17));

            ft2 =  (boundary[gpu_scalar_index(idx-1, idy, idz)] == 4)?0:f1[gpu_fieldn_index(idx-1, idy, idz,2)];//0.5*(on1-6*ws*(-lat_ux));//2*f1[gpu_fieldn_index(idx-1, idy, idz, 2)] - f1[gpu_fieldn_index(idx-2, idy, idz, 2)]; 
            ft8 =  (boundary[gpu_scalar_index(idx-1, ym1, idz)] == 4)?f1[gpu_fieldn_index(idx-1, idy, idz,7)]:f1[gpu_fieldn_index(idx-1, ym1, idz,8)];//0.5*(on7-6*wd*(-lat_ux-lat_uy)-(-lat_uy)/3.0+(on4+on12+on18)/2.0);//2*f1[gpu_fieldn_index(idx-1, idy, idz, 8)] - f1[gpu_fieldn_index(idx-2, idy, idz, 8)]; 
            ft10 = (boundary[gpu_scalar_index(idx-1, idy, zm1)] == 4)?f1[gpu_fieldn_index(idx-1, idy, idz,9)]:f1[gpu_fieldn_index(idx-1, idy, zm1,10)];//0.5*(on9-6*wd*(-lat_ux-lat_uz)-(-lat_uz)/3.0+(on6+on12+on17)/2.0);//2*f1[gpu_fieldn_index(idx-1, idy, idz, 10)] - f1[gpu_fieldn_index(idx-2, idy, idz, 10)];
            ft14 = (boundary[gpu_scalar_index(idx-1, yp1, idz)] == 4)?f1[gpu_fieldn_index(idx-1, idy, idz,13)]:f1[gpu_fieldn_index(idx-1, yp1, idz,14)];//0.5*(on13-6*wd*(-lat_ux+lat_uy)-(lat_uy)/3.0+(on3+on11+on17)/2.0);//2*f1[gpu_fieldn_index(idx-1, idy, idz, 14)] - f1[gpu_fieldn_index(idx-2, idy, idz, 14)]; 
            ft16 = (boundary[gpu_scalar_index(idx-1, idy, zp1)] == 4)?f1[gpu_fieldn_index(idx-1, idy, idz,15)]:f1[gpu_fieldn_index(idx-1, idy, zp1,16)];//0.5*(on15-6*wd*(-lat_ux+lat_uz)-(lat_uz)/3.0+(on5+on11+on18)/2.0);//2*f1[gpu_fieldn_index(idx-1, idy, idz, 16)] - f1[gpu_fieldn_index(idx-2, idy, idz, 16)]; 
        }
    }

    lat_rho = (bound==4)?0:ft0+ft1+ft2+ft3+ft4+ft5+ft6+ft7+ft8+ft9+ft10+ft11+ft12+ft13+ft14+ft15+ft16+ft17+ft18;
    rhoinv = (bound==4)?0:min(10000.0, 1.0/(lat_rho));

    lat_ux = (bound==4)?0:rhoinv*(ft1+ft7+ft9+ft13+ft15-(ft2+ft8+ft10+ft14+ft16));
    lat_uy = (bound==4)?0:rhoinv*(ft3+ft7+ft11+ft14+ft17-(ft4+ft8+ft12+ft13+ft18));
    lat_uz = (bound==4)?0:rhoinv*(ft5+ft9+ft11+ft16+ft18-(ft6+ft10+ft12+ft15+ft17));

    float tw0r = tauinv*w0*lat_rho; //   w[0]*rho/tau 
    float twsr = tauinv*ws*lat_rho; // w[1-4]*rho/tau
    float twdr = tauinv*wd*lat_rho; // w[5-8]*rho/tau
    float omusq = 1.0 - 1.5*(lat_ux*lat_ux+lat_uy*lat_uy+lat_uz*lat_uz); // 1-(3/2)u.u

    if(save)
    {
        rho[gpu_scalar_index(idx, idy, idz)] = lat_rho;
        ux[gpu_scalar_index(idx, idy, idz)] = lat_ux;
        uy[gpu_scalar_index(idx, idy, idz)] = lat_uy;
        uz[gpu_scalar_index(idx, idy, idz)] = lat_uz;
    }

    float tux = 3.0*lat_ux;
    float tuy = 3.0*lat_uy;
    float tuz = 3.0*lat_uz;

    f0[gpu_field0_index(idx,idy,idz)] = tw0r*(omusq);
    
    collide(f2,idx, idy, idz, omusq, tux, tuy, tuz, twsr, twdr);

}

// __device__ void stream_enforce_boundary(float *f1, float* f0, short* boundary, int idx, int idy, int idz)
// {
//     short bound = boundary[gpu_scalar_index(idx, idy, idz)];

//     unsigned int xp1 = (idx+1)%NX; //front in X direction
//     unsigned int yp1 = (idy+1)%NY; //front in Y direction
//     unsigned int zp1 = (idz+1)%NZ; //front in Z direction
//     unsigned int xm1 = (NX+idx-1)%NX; //back in X direction
//     unsigned int ym1 = (NY+idy-1)%NY; //back in Y direction
//     unsigned int zm1 = (NZ+idz-1)%NZ; //back in Z direction

//     float ft0 = f0[gpu_field0_index(idx,idy,idz)];

//     // load populations from adjacent nodes
//     float ft1, ft2, ft3, ft4, ft5, ft6, ft7, ft8, ft9;
//     float ft10, ft11, ft12, ft13, ft14, ft15, ft16, ft17, ft18;
    
//     ft1 = (boundary[gpu_scalar_index(xm1, idy, idz)] == 4)?f1[gpu_fieldn_index(idx, idy, idz, 2)]:f1[gpu_fieldn_index(xm1, idy, idz, 1)];
//     ft2 = (boundary[gpu_scalar_index(xp1, idy, idz)] == 4)?f1[gpu_fieldn_index(idx, idy, idz, 1)]:f1[gpu_fieldn_index(xp1, idy, idz, 2)];
//     ft3 = (boundary[gpu_scalar_index(idx, ym1, idz)] == 4)?f1[gpu_fieldn_index(idx, idy, idz, 4)]:f1[gpu_fieldn_index(idx, ym1, idz, 3)];
//     ft4 = (boundary[gpu_scalar_index(idx, yp1, idz)] == 4)?f1[gpu_fieldn_index(idx, idy, idz, 3)]:f1[gpu_fieldn_index(idx, yp1, idz, 4)];
//     ft5 = (boundary[gpu_scalar_index(idx, idy, zm1)] == 4)?f1[gpu_fieldn_index(idx, idy, idz, 6)]:f1[gpu_fieldn_index(idx, idy, zm1, 5)];
//     ft6 = (boundary[gpu_scalar_index(idx, idy, zp1)] == 4)?f1[gpu_fieldn_index(idx, idy, idz, 5)]:f1[gpu_fieldn_index(idx, idy, zp1, 6)];
//     ft7 = (boundary[gpu_scalar_index(xm1, ym1, idz)] == 4)?f1[gpu_fieldn_index(idx, idy, idz, 8)]:f1[gpu_fieldn_index(xm1, ym1, idz, 7)];
//     ft8 = (boundary[gpu_scalar_index(xp1, yp1, idz)] == 4)?f1[gpu_fieldn_index(idx, idy, idz, 7)]:f1[gpu_fieldn_index(xp1, yp1, idz, 8)];
//     ft9 = (boundary[gpu_scalar_index(xm1, idy, zm1)] == 4)?f1[gpu_fieldn_index(idx, idy, idz, 10)]:f1[gpu_fieldn_index(xm1, idy, zm1, 9)];
//     ft10 = (boundary[gpu_scalar_index(xp1, idy, zp1)] == 4)?f1[gpu_fieldn_index(idx, idy, idz, 9)]:f1[gpu_fieldn_index(xp1, idy, zp1, 10)];
//     ft11 = (boundary[gpu_scalar_index(idx, ym1, zm1)] == 4)?f1[gpu_fieldn_index(idx, idy, idz, 12)]:f1[gpu_fieldn_index(idx, ym1, zm1, 11)];
//     ft12 = (boundary[gpu_scalar_index(idx, yp1, zp1)] == 4)?f1[gpu_fieldn_index(idx, idy, idz, 11)]:f1[gpu_fieldn_index(idx, yp1, zp1, 12)];
//     ft13 = (boundary[gpu_scalar_index(xm1, yp1, idz)] == 4)?f1[gpu_fieldn_index(idx, idy, idz, 14)]:f1[gpu_fieldn_index(xm1, yp1, idz, 13)];
//     ft14 = (boundary[gpu_scalar_index(xp1, ym1, idz)] == 4)?f1[gpu_fieldn_index(idx, idy, idz, 13)]:f1[gpu_fieldn_index(xp1, ym1, idz, 14)];
//     ft15 = (boundary[gpu_scalar_index(xm1, idy, zp1)] == 4)?f1[gpu_fieldn_index(idx, idy, idz, 16)]:f1[gpu_fieldn_index(xm1, idy, zp1, 15)];
//     ft16 = (boundary[gpu_scalar_index(xp1, idy, zm1)] == 4)?f1[gpu_fieldn_index(idx, idy, idz, 15)]:f1[gpu_fieldn_index(xp1, idy, zm1, 16)];
//     ft17 = (boundary[gpu_scalar_index(idx, ym1, zp1)] == 4)?f1[gpu_fieldn_index(idx, idy, idz, 18)]:f1[gpu_fieldn_index(idx, ym1, zp1, 17)];
//     ft18 = (boundary[gpu_scalar_index(idx, yp1, zm1)] == 4)?f1[gpu_fieldn_index(idx, idy, idz, 17)]:f1[gpu_fieldn_index(idx, yp1, zm1, 18)];

//     if (bound == 2)
//     {
//         ft2 = f1[gpu_fieldn_index(idx, idy, idz, 1)] + 6*ws;
//         ft10 = f1[gpu_fieldn_index(idx, idy, idz, 9)] + 6*wd;
//         ft16 = f1[gpu_fieldn_index(idx, idy, idz, 15)] + 6*wd;
//         ft14 = f1[gpu_fieldn_index(idx, idy, idz, 13)] + 6*wd;
//         ft8 = f1[gpu_fieldn_index(idx, idy, idz, 7)] + 6*wd;
//     }

//     if(bound == 7)
//     {
//         ft2 = 2*f1[gpu_fieldn_index(idx-1, idy, idz, 1)] - f1[gpu_fieldn_index(idx-2, idy, idz, 1)]; 
//         ft8 = 2*f1[gpu_fieldn_index(idx-1, idy, idz, 8)] - f1[gpu_fieldn_index(idx-2, idy, idz, 8)]; 
//         ft10 = 2*f1[gpu_fieldn_index(idx-1, idy, idz, 10)] - f1[gpu_fieldn_index(idx-2, idy, idz, 10)];
//         ft14 = 2*f1[gpu_fieldn_index(idx-1, idy, idz, 14)] - f1[gpu_fieldn_index(idx-2, idy, idz, 14)]; 
//         ft16 = 2*f1[gpu_fieldn_index(idx-1, idy, idz, 16)] - f1[gpu_fieldn_index(idx-2, idy, idz, 16)]; 
//     }
   
//     f1[gpu_fieldn_index(idx, idy, idz,1)]=ft1;
//     f1[gpu_fieldn_index(idx, idy, idz,2)]=ft2;
//     f1[gpu_fieldn_index(idx, idy, idz,3)]=ft3;
//     f1[gpu_fieldn_index(idx, idy, idz,4)]=ft4;
//     f1[gpu_fieldn_index(idx, idy, idz,5)]=ft5;
//     f1[gpu_fieldn_index(idx, idy, idz,6)]=ft6;
//     f1[gpu_fieldn_index(idx, idy, idz,7)]=ft7;
//     f1[gpu_fieldn_index(idx, idy, idz,8)]=ft8;
//     f1[gpu_fieldn_index(idx, idy, idz,9)]=ft9;
//     f1[gpu_fieldn_index(idx, idy, idz,10)]=ft10;
//     f1[gpu_fieldn_index(idx, idy, idz,11)]=ft11;
//     f1[gpu_fieldn_index(idx, idy, idz,12)]=ft12;
//     f1[gpu_fieldn_index(idx, idy, idz,13)]=ft13;
//     f1[gpu_fieldn_index(idx, idy, idz,14)]=ft14;
//     f1[gpu_fieldn_index(idx, idy, idz,15)]=ft15;
//     f1[gpu_fieldn_index(idx, idy, idz,16)]=ft16;
//     f1[gpu_fieldn_index(idx, idy, idz,17)]=ft17;
//     f1[gpu_fieldn_index(idx, idy, idz,18)]=ft18;
// }
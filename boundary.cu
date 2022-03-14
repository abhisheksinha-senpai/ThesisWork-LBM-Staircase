#include "boundary.h"
#include "lbm.h"
#include "utilities.h"

using namespace std;
bool *cpu_boundary;
bool *gpu_boundary;
short *cpu_normals;
short *gpu_normals;

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
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idz = threadIdx.z +  blockIdx.y * blockDim.z;

    unsigned int sidx = gpu_scalar_index(idx, idy, idz);
    rho[sidx] = (boundary[sidx] == 2) 1.0? 0.0;
    ux[sidx] = 1.0;
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
                    if(j==0)
                    {
                        if(i<NY/4 || i>NY/2)
                        {
                            cpu_boundary[k*NX*NY + i*NX +j] = 4;//wall;
                            cpu_normals[k*NX*NY + i*NX +j] = 0;
                        }
                        else
                        {
                            cpu_boundary[k*NX*NY + i*NX +j] = 2;//inlet;
                            cpu_normals[k*NX*NY + i*NX +j] = 0b00000001;
                        }
                    }
                    else
                    {
                        if(i<NY/4 || i>NY/2)
                        {
                            cpu_boundary[k*NX*NY + i*NX +j] = 4;//wall;
                        }
                        else if(i == NY/4 || i == NY/2)
                        {
                            cpu_boundary[k*NX*NY + i*NX +j] = 3;//boundary;
                            cpu_normals[k*NX*NY + i*NX +j] = (i==NY/4)?0b00000010:0b00100010;
                        }
                        else
                        {
                            cpu_boundary[k*NX*NY + i*NX +j] = 1;//fluid
                        }
                   }
                }
                else if(j<NX/4)
                {
                    if(j==NX/8)
                    {
                        if(i>(NY-NY/4) || i<NY/4)
                        {
                            cpu_boundary[k*NX*NY + i*NX +j] = 4;//wall
                            cpu_normals[k*NX*NY + i*NX +j] = 0;
                        }
                        else if(i==(NY-NY/4)|| i == NY/2)
                        {
                            cpu_boundary[k*NX*NY + i*NX +j] = 3;//boundary
                            if(i==(NY-NY/4))
                                cpu_normals[k*NX*NY + i*NX +j] = 0b01000011;
                            else if(i == NY-NY/4)
                                cpu_normals[k*NX*NY + i*NX +j] = 0b01000011;
                            else
                                cpu_normals[k*NX*NY + i*NX +j] = 0b00000010;
                        }
                        else if(i>NY/2 && i<(NY-NY/4))
                        {
                            cpu_boundary[k*NX*NY + i*NX +j] = 3;//boundary
                            cpu_normals[k*NX*NY + i*NX +j] = 0b00000001;
                        }
                        else
                        {
                            cpu_boundary[k*NX*NY + i*NX +j] = 3;//fluid
                            cpu_normals[k*NX*NY + i*NX +j] = 0;
                        }  
                    }
                    else
                    {
                        if(i>(NY-NY/4) || i<NY/4)
                        {
                            cpu_boundary[k*NX*NY + i*NX +j] = 4;//wall
                            cpu_normals[k*NX*NY + i*NX +j] = 0;
                        }
                        else if(i==(NY-NY/4) || i==NY/4)
                        {
                            cpu_boundary[k*NX*NY + i*NX +j] = 3;//boundary
                            cpu_normals[k*NX*NY + i*NX +j] = (i==NY/4)?0b00000010:0b01000010;
                        }
                        else
                        {
                            cpu_boundary[k*NX*NY + i*NX +j] = 1;//fluid
                            cpu_normals[k*NX*NY + i*NX +j] = 0;
                        }
                    }
                }
                else if(j<(NX/2 - NX/4))
                {
                    if(j==NX/4)
                    {
                        if(i<NY/4)
                        {
                            cpu_boundary[k*NX*NY + i*NX +j] = 4;//wall
                            cpu_normals[k*NX*NY + i*NX +j] = 0;
                        }
                        else if(i==(NY-NY/4) || i == NY/4 || i == NY/2 || i == NY - 1)
                        {
                            cpu_boundary[k*NX*NY + i*NX +j] = 3;//boundary
                            if(i==(NY-NY/4))
                                cpu_normals[k*NX*NY + i*NX +j] = 0b01000011;
                            else if(i == NY/4)
                                cpu_normals[k*NX*NY + i*NX +j] = 0b00100011;
                            else if(i == NY-1)
                                cpu_normals[k*NX*NY + i*NX +j] = 0b01000011;
                            else
                                cpu_normals[k*NX*NY + i*NX +j] = 0b00100011;
                        }
                        else if(i>(NY-NY/4) && i<(NY))
                        {
                            cpu_boundary[k*NX*NY + i*NX +j] = 3;//boundary
                            cpu_normals[k*NX*NY + i*NX +j] = 0b00000001;
                        }
                        else if(i<NY/2 && i>NY/4)
                        {
                            cpu_boundary[k*NX*NY + i*NX +j] = 3;//boundary
                            cpu_normals[k*NX*NY + i*NX +j] = 0b00100001;
                        }
                        else
                        {
                            cpu_boundary[k*NX*NY + i*NX +j] = 1;//fluid
                            cpu_normals[k*NX*NY + i*NX +j] = 0;
                        }  
                    }
                    else
                    {
                        if(i<NY/2)
                        {
                            cpu_boundary[k*NX*NY + i*NX +j] = 4;//wall
                            cpu_normals[k*NX*NY + i*NX +j] = 0;
                        }
                        else if(i==NY/2 || i == NY-1)
                        {
                            cpu_boundary[k*NX*NY + i*NX +j] = 3;//boundary
                            if(i== NY-1)
                                cpu_normals[k*NX*NY + i*NX +j] = 0b01000010;
                            else
                                cpu_normals[k*NX*NY + i*NX +j] = 0b00000010;
                        }
                        else
                        {
                            cpu_boundary[k*NX*NY + i*NX +j] = 1;//fluid
                            cpu_normals[k*NX*NY + i*NX +j] = 0;
                        }
                    }
                }
                else if(j<(NX/2))
                {
                    if(j==(NX/2 - NX/4))
                    {
                        if(i<NY/2)
                        {
                            cpu_boundary[k*NX*NY + i*NX +j] = 4;//wall
                            cpu_normals[k*NX*NY + i*NX +j] = 0;
                        }
                        else if(i<(NY - NY/4))
                        {
                            cpu_boundary[k*NX*NY + i*NX +j] = 3;//boundary
                            if(i==NY/2)
                                cpu_normals[k*NX*NY + i*NX +j] = 0b00100011;
                            else
                                cpu_normals[k*NX*NY + i*NX +j] = 0b00100001;
                            
                        }
                        else if(i == (NY-NY/4))
                        {
                            cpu_boundary[k*NX*NY + i*NX +j] = 3;//boundary
                            cpu_normals[k*NX*NY + i*NX +j] = 0b00100011;
                        }
                        else if(i==NY-1)
                        {
                            cpu_boundary[k*NX*NY + i*NX +j] = 3;//boundary
                            cpu_normals[k*NX*NY + i*NX +j] = 0b01000010;
                        }
                        else
                        {
                            cpu_boundary[k*NX*NY + i*NX +j] = 1;//fluid
                            cpu_normals[k*NX*NY + i*NX +j] = 0;
                        }  
                    }
                    else
                    {
                        if(i<(NY - NY/4))
                        {
                            cpu_boundary[k*NX*NY + i*NX +j] = 4;//wall
                            cpu_normals[k*NX*NY + i*NX +j] = 0;
                        }
                        else if(i==(NY - NY/4) || i == NY-1)
                        {
                            cpu_boundary[k*NX*NY + i*NX +j] = 3;//boundary
                            if(i== NY-1)
                                cpu_normals[k*NX*NY + i*NX +j] = 0b01000010;
                            else
                                cpu_normals[k*NX*NY + i*NX +j] = 0b00000010;
                        }
                        else
                        {
                            cpu_boundary[k*NX*NY + i*NX +j] = 1;//fluid
                            cpu_normals[k*NX*NY + i*NX +j] = 0;
                        }
                    }
                }
                else if(j == NX/2)
                {
                    if(i<=(NY - NY/4) || i == NY-1)
                    {
                        cpu_boundary[k*NX*NY + i*NX +j] = 3;//boundary
                        if(i == NY-1)
                            cpu_normals[k*NX*NY + i*NX +j] = 0b01000010;
                        else if( i == (NY - NY/4))
                            cpu_normals[k*NX*NY + i*NX +j] = 0b00100011;
                        else if(i == 0)
                            cpu_normals[k*NX*NY + i*NX +j] = 0b00100011;
                    }
                    else
                    {
                        cpu_boundary[k*NX*NY + i*NX +j] = 1;//fluid
                        cpu_normals[k*NX*NY + i*NX +j] = 0;
                    }
                }
                else
                {
                    if(i == 0 || i == NY -1)
                    {
                        cpu_boundary[k*NX*NY + i*NX +j] = 3;
                        if(i == NY-1)
                            cpu_normals[k*NX*NY + i*NX +j] = 0b01000010;
                        else
                            cpu_normals[k*NX*NY + i*NX +j] = 0b00000010;
                    }
                    else
                    {
                        cpu_boundary[k*NX*NY + i*NX +j] = 1;
                        cpu_normals[k*NX*NY + i*NX +j] = 0;
                    }
                }            
            }
        }
    }
}

__device__ void enforce_boundary(float *f1, int4* boundary, int idx, int idy, int idz)
{
    int4 bound = boundary[gpu_scalar_index(idx, idy, idz)];

    unsigned int xp1 = (idx+1)%NX; //front in X direction
    unsigned int yp1 = (idy+1)%NY; //front in Y direction
    unsigned int zp1 = (idz+1)%NZ; //front in Z direction
    unsigned int xm1 = (NX+idx-1)%NX; //back in X direction
    unsigned int ym1 = (NY+idy-1)%NY; //back in Y direction
    unsigned int zm1 = (NZ+idz-1)%NZ; //back in Z direction

    float ft0 = f0[gpu_field0_index(idx,idy,idz)];

    // load populations from adjacent nodes
    float pos_src = gpu_fieldn_index(xm1, idy, idz, 1)
    float dst = gpu_fieldn_index(xm1, idy, idz, 1)
    f1[] = f1[gpu_fieldn_index(xm1, idy, idz, 1)];
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
}
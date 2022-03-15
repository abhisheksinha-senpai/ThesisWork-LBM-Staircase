#include "boundary.h"
#include "lbm.h"
#include "utilities.h"

using namespace std;
short *cpu_boundary;
short *gpu_boundary;
short *cpu_normals;
short *gpu_normals;

__host__ void cpu_field_Initialization()
{
    defineBoundary();
    checkCudaErrors(cudaMemcpy(gpu_boundary, cpu_boundary, mem_size_bound,  cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(gpu_normals, cpu_normals, mem_size_normal,  cudaMemcpyHostToDevice));
    // blocks in grid
    dim3  grid(NX/nThreads, NY, NZ);
    // threads in block
    dim3  threads(nThreads, 1, 1);

    gpu_field_Initialization<<< grid, threads >>>(gpu_boundary, rho_gpu, ux_gpu, uy_gpu, uz_gpu, f1_gpu, f0_gpu);
    getLastCudaError("\ngpu_field_Initialization kernel error\n");
}

__global__ void gpu_field_Initialization(short *boundary, float *rho, float *ux, float *uy, float *uz, float* field, float* field0)
{
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int z = threadIdx.z +  blockIdx.z * blockDim.z;

    unsigned int sidx = gpu_scalar_index(x, y, z);
    rho[sidx] = 0.0;
    ux[sidx] = 0.0;
    uy[sidx] = 0.0;
    uz[sidx] = 0.0;
    field[gpu_fieldn_index(x,y,z,1)]=0.0;
    field[gpu_fieldn_index(x,y,z,2)]=0.0;
    field[gpu_fieldn_index(x,y,z,3)]=0.0;
    field[gpu_fieldn_index(x,y,z,4)]=0.0;
    field[gpu_fieldn_index(x,y,z,5)]=0.0;
    field[gpu_fieldn_index(x,y,z,6)]=0.0;
    field[gpu_fieldn_index(x,y,z,7)]=0.0;
    field[gpu_fieldn_index(x,y,z,8)]=0.0;
    field[gpu_fieldn_index(x,y,z,9)]=0.0;
    field[gpu_fieldn_index(x,y,z,10)]=0.0;
    field[gpu_fieldn_index(x,y,z,11)]=0.0;
    field[gpu_fieldn_index(x,y,z,12)]=0.0;
    field[gpu_fieldn_index(x,y,z,13)]=0.0;
    field[gpu_fieldn_index(x,y,z,14)]=0.0;
    field[gpu_fieldn_index(x,y,z,15)]=0.0;
    field[gpu_fieldn_index(x,y,z,16)]=0.0;
    field[gpu_fieldn_index(x,y,z,17)]=0.0;
    field[gpu_fieldn_index(x,y,z,18)]=0.0;
    field0[sidx] = 0.0;


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
                        else if(i==(NY-NY/4)|| i == NY/2 || i== NY/4)
                        {
                            if(i==(NY-NY/4))
                            {
                                cpu_normals[k*NX*NY + i*NX +j] = 0b01000011;
                                cpu_boundary[k*NX*NY + i*NX +j] = 5;
                            }
                            else if(i == NY/2)
                            {
                                cpu_normals[k*NX*NY + i*NX +j] = 0b01000011;
                                cpu_boundary[k*NX*NY + i*NX +j] = 6;
                            }
                            else
                            {
                                cpu_normals[k*NX*NY + i*NX +j] = 0b00000010;
                                cpu_boundary[k*NX*NY + i*NX +j] = 3;
                            }
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
                            if(i==(NY-NY/4))
                            {
                                cpu_normals[k*NX*NY + i*NX +j] = 0b01000011;
                                cpu_boundary[k*NX*NY + i*NX +j] = 6;
                            }
                            else if(i == NY/4)
                            {
                                cpu_normals[k*NX*NY + i*NX +j] = 0b00100011;
                                cpu_boundary[k*NX*NY + i*NX +j] = 5;
                            }
                            else if(i == NY-1)
                            {
                                cpu_normals[k*NX*NY + i*NX +j] = 0b01000011;
                                cpu_boundary[k*NX*NY + i*NX +j] = 5;
                            }
                            else
                            {
                                cpu_normals[k*NX*NY + i*NX +j] = 0b00100011;
                                cpu_boundary[k*NX*NY + i*NX +j] = 6;
                            }
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
                            {
                                cpu_boundary[k*NX*NY + i*NX +j] = 5;
                                cpu_normals[k*NX*NY + i*NX +j] = 0b00100011;
                            }
                            else
                                cpu_normals[k*NX*NY + i*NX +j] = 0b00100001;
                            
                        }
                        else if(i == (NY-NY/4))
                        {
                            cpu_boundary[k*NX*NY + i*NX +j] = 6;//boundary
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
                        {
                            cpu_boundary[k*NX*NY + i*NX +j] = 6;
                            cpu_normals[k*NX*NY + i*NX +j] = 0b00100011;
                        }
                        else if(i == 0)
                        {
                            cpu_boundary[k*NX*NY + i*NX +j] = 5;
                            cpu_normals[k*NX*NY + i*NX +j] = 0b00100011;
                        }
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
                        cpu_boundary[k*NX*NY + i*NX +j] = 4;
                        cpu_normals[k*NX*NY + i*NX +j] = 0;
                    }
                    else if(i == 1 || i == NY -2)
                    {
                        cpu_boundary[k*NX*NY + i*NX +j] = 3;
                        if(i == NY-2)
                            cpu_normals[k*NX*NY + i*NX +j] = 0b01000010;
                        else
                            cpu_normals[k*NX*NY + i*NX +j] = 0b00000010;
                    }
                    else
                    {
                        cpu_boundary[k*NX*NY + i*NX +j] = 1;
                        cpu_normals[k*NX*NY + i*NX +j] = 0;
                    }
                    if(j == NX-1)
                    {
                        cpu_boundary[k*NX*NY + i*NX +j] = 7;
                        cpu_normals[k*NX*NY + i*NX +j] = 0;
                    }
                }            
            }
        }
    }
}

// __device__ void enforce_boundary(float *f1, float* f0, short* boundary, short* normals, int idx, int idy, int idz)
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
//         ft2 = f1[gpu_fieldn_index(idx, idy, idz, 1)] + 6*wd;
//         ft10 = f1[gpu_fieldn_index(idx, idy, idz, 9)] + 6*wd;
//         ft16 = f1[gpu_fieldn_index(idx, idy, idz, 15)] + 6*wd;
//         ft14 = f1[gpu_fieldn_index(idx, idy, idz, 13)] + 6*wd;
//         ft8 = f1[gpu_fieldn_index(idx, idy, idz, 7)] + 6*wd;
//     }

//     if(bound == 7)
//     {
//         ft2 = 2*f1[gpu_fieldn_index(idx-1, idy, idz, 1)] - f1[gpu_fieldn_index(idx-2, idy, idz, 2)]; 
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
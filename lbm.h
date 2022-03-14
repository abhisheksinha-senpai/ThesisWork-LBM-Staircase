#ifndef LBM_H_
#define LBM_H__

#include "utilities.h"

const float w0 = 12.0/36.0;  // zero weight
const float ws = 2.0/36.0;  // adjacent weight
const float wd = 1.0/36.0; // diagonal weight

const float nu = 1.0/6.0;
const float tau = 3.0*nu+0.5;

const unsigned int scale = 2;
const unsigned int NX = 512;
const unsigned int NY = 256;
const unsigned int NZ = 32;
const unsigned int NSTEPS = 1*scale*scale;

const unsigned int ndir = 19;
const unsigned int mem_size_0dir   = sizeof(float)*NX*NY*NZ;
const unsigned int mem_size_n0dir  = sizeof(float)*NX*NY*NZ*(ndir-1);
const unsigned int mem_size_scalar = sizeof(float)*NX*NY*NZ;
const unsigned int mem_size_bound= sizeof(bool)*NX*NY*NZ;
const unsigned int mem_size_normal= sizeof(short)*NX*NY*NZ;

extern unsigned int mem_size_props;
extern float *f0_gpu,*f1_gpu,*f2_gpu;
extern float *rho_gpu,*ux_gpu,*uy_gpu, *uz_gpu;
extern float *prop_gpu;
extern float *scalar_host;
extern bool *cpu_boundary;
extern bool *gpu_boundary;
extern short *cpu_normals;
extern short *gpu_normals;

__device__ __forceinline__ unsigned int gpu_field0_index(unsigned int x, unsigned int y, unsigned int z)
{
    return (z*(NX*NY) + NX*y + x);
}

__device__ __forceinline__ unsigned int gpu_scalar_index(unsigned int x, unsigned int y, unsigned int z)
{
    return (z*(NX*NY) + NX*y + x);
}

__device__ __forceinline__ unsigned int gpu_fieldn_index(unsigned int x, unsigned int y, unsigned int z, unsigned int d)
{
    return ((d-1)*NX*NY*NZ + z*(NX*NY) + NX*y + x);
}

__host__ void cpu_equi_Initialization();

__global__ void gpu_equi_Initialization(float *f0, float *f1, float *rho, 
float *ux, float *uy, float *uz);

__host__ void cpu_stream_collide();

__global__ void gpu_stream_collide(int4* boundary, float *f0, float *f1, float *f2, float *rho
, float *ux, float *uy, float *uz);
#endif
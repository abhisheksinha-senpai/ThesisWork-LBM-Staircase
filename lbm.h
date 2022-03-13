#ifndef LBM_H_
#define LBM_H__

#include "utilities.h"

const float w0 = 12.0/36.0;  // zero weight
const float ws = 2.0/18.0;  // adjacent weight
const float wd = 1.0/36.0; // diagonal weight

const unsigned int scale = 2;
const unsigned int NX = 512;
const unsigned int NY = 256;
const unsigned int NZ = 32;
const unsigned int NSTEPS = 200*scale*scale;

const unsigned int ndir = 19;
const unsigned int mem_size_0dir   = sizeof(float)*NX*NY*NZ;
const unsigned int mem_size_n0dir  = sizeof(float)*NX*NY*NZ*(ndir-1);
const unsigned int mem_size_scalar = sizeof(float)*NX*NY*NZ;
const unsigned int mem_size_bound= sizeof(bool)*NX*NY*NZ;

extern unsigned int mem_size_props;
extern float *f0_gpu,*f1_gpu,*f2_gpu;
extern float *rho_gpu,*ux_gpu,*uy_gpu, *uz_gpu;
extern float *prop_gpu;
extern float *scalar_host;
extern bool *cpu_boundary;
extern bool *gpu_boundary;

__device__ __forceinline__ unsigned int gpu_field0_index(unsigned int x, unsigned int y, unsigned int z)
{
    return NX*y+x;
}

__device__ __forceinline__ unsigned int gpu_scalar_index(unsigned int x, unsigned int y, unsigned int z)
{
    return NX*y+x;
}

__device__ __forceinline__ unsigned int gpu_fieldn_index(unsigned int x, unsigned int y, unsigned int z, unsigned int d)
{
    return (NX*(NY*(d-1)+y)+x);
}

__host__ void cpu_equi_Initialization();
__global__ void gpu_equi_Initialization(bool *boundary, float *rho, float *ux, float *uy, float *uz);

#endif
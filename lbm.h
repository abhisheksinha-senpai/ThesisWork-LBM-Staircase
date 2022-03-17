#ifndef LBM_H_
#define LBM_H__

const float w0 = 12.0/36.0;  // zero weight
const float ws = 2.0/36.0;  // adjacent weight
const float wd = 1.0/36.0; // diagonal weight


const unsigned int length_scale = 8;
const unsigned int time_scale = 10;
const unsigned int NX = 64*length_scale;
const unsigned int NY = NX/4;
const unsigned int NZ = 2;

const float nu = 1.0/6.0;
const float tau = 3.0*nu+0.5;

const unsigned int NSTEPS = 20*time_scale*time_scale;
const unsigned int NSAVE  =  0.5*time_scale*time_scale;

const unsigned int ndir = 19;
const unsigned int mem_size_0dir   = sizeof(float)*NX*NY*NZ;
const unsigned int mem_size_n0dir  = sizeof(float)*NX*NY*NZ*(ndir-1);
const unsigned int mem_size_scalar = sizeof(float)*NX*NY*NZ;
const unsigned int mem_size_bound = sizeof(short)*NX*NY*NZ;

extern unsigned int mem_size_props;
extern float *f0_gpu,*f1_gpu,*f2_gpu;
extern float *rho_gpu,*ux_gpu,*uy_gpu, *uz_gpu;
extern float *prop_gpu;
extern float *scalar_host;


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

__host__ void cpu_stream_collide(bool save);

__global__ void gpu_stream_collide(short* boundary, float *f0, float *f1, float *f2, float *rho
, float *ux, float *uy, float *uz, bool save);

__device__ void collide(float *field,int x, int y, int z,
     float omusq, float tux, float tuy, float tuz, float wsr, float wsd, float tau, float delT);
#endif
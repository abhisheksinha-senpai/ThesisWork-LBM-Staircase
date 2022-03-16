#ifndef BOUNDARY_H_
#define BOUNDARY_H_

extern short *cpu_boundary;
extern short *gpu_boundary;
extern short *cpu_normals;
extern short *gpu_normals;

__host__ void cpu_field_Initialization();
__host__ void defineBoundary();
__global__ void gpu_field_Initialization(short *boundary, float *rho, float *ux, float *uy, float *uz, float* field, float *field0);
__device__ void stream_enforce_boundary(float *f1, float* f0, short* boundary, int idx, int idy, int idz);

#endif
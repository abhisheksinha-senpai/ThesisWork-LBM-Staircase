#ifndef BOUNDARY_H_
#define BOUNDARY_H_

#include "utilities.h"

__host__ void cpu_field_Initialization();
__host__ void defineBoundary();
__global__ void gpu_field_Initialization(bool *boundary, float *rho, float *ux, float *uy, float *uz);
__device__ void enforce_boundary(float *f1, int4* boundary, int idx, int idy, int idz);

#endif
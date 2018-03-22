#include "amul.h"
// absolute values of a field

extern "C" __global__ void
MSliceMul(float* __restrict__ dst, float* __restrict__ src, float* __restrict__ MSliceSrc, float MSliceMultiplier, int Nx, int Ny, int Nz) {

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	if(ix >= Nx || iy >=  Ny || iz >= Nz) {
		return;
	}

	int idx = ((iz)*(Ny) + (iy))*(Nx) + (ix);

	float MSliceVal = amul(MSliceSrc, MSliceMultiplier, idx);

	dst[idx] = src[idx]*MSliceVal;
}


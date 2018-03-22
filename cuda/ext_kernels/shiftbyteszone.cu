#include <stdint.h>
#include "stencil.h"

// shift dst by shx cells (positive or negative) along X-axis.
// new edge value is clampL at left edge or clampR at right edge.
extern "C" __global__ void
shiftbyteszone(uint8_t* __restrict__  dst, uint8_t* __restrict__  src,
           int Nx,  int Ny,  int Nz, int shx, int shy, int shz, uint8_t clamp, float* __restrict__ zoneGeom) {

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	if(ix >= Nx || iy >= Ny || iz >= Nz) {
		return;
	}

	int sI = index(ix, iy, iz, Nx, Ny, Nz);
	
	int ix2 = (ix+shx+Nx)%Nx;
	int iy2 = (iy+shy+Ny)%Ny;
	int iz2 = (iz+shz+Nz)%Nz;

	int dI = index(ix2, iy2, iz2, Nx, Ny, Nz);

	if(zoneGeom[sI] == 0 && zoneGeom[dI] == 0)
	{
		return;
	}


	dst[dI] = src[sI];
}


#include "stencil.h"

// Copy src (size S) to dst (size D)
extern "C" __global__ void
copyunpadGeom(float* __restrict__  dst, int Dx, int Dy, int Dz,
          float* __restrict__  src, int Sx, int Sy, int Sz, int shift, int Periodic, float* geometry, int includeOrExclude) {

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	if(ix >= Dx || iy >= Dy || iz >= Dz) {
		return;
	}

	int dI = index(ix, iy, iz, Dx, Dy, Dz);

	if(includeOrExclude == 1 && geometry[dI] == 0) //When including only the geometry, we return outside of it.
		return;
	else if(includeOrExclude == 0 && geometry[dI] != 0) //we exclude the geometry, so return if we're inside of it
		return;

	if(Periodic == 1 || (ix+shift >= 0 && ix < Dx+shift && iy+shift >= 0 && iy < Dy+shift && iz+shift >= 0 && iz < Dz+shift)) //Periodic or in the area inside dst
		dst[dI] = src[index((ix+shift+Sx)%Sx, (iy+shift+Sy)%Sy, (iz+shift+Sz)%Sz, Sx, Sy, Sz)];
	else
		dst[dI] = 0;
}


#include "stencil.h"
#include "amul.h"
#include "constants.h"
#include <stdint.h>

// Copy src (size S, smaller) into dst (size D, larger),
// and multiply by Bsat as defined in regions.
extern "C" __global__ void
copypadmulGeom(float* __restrict__ dst, int Dx, int Dy, int Dz,
           float* __restrict__ src, float* __restrict__ vol, int Sx, int Sy, int Sz,
           float* __restrict__ Ms_, float Ms_mul, float* geometry, int useBSat, int includeOrExclude) {

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	if(ix >= Sx || iy >= Sy || iz >= Sz)
	{
		return;
	}

	int sI = index(ix, iy, iz, Sx, Sy, Sz);  // source index
	

	if(includeOrExclude == 1 && geometry[sI] == 0) //When including only the geometry, we return outside of it.
		return;
	else if(includeOrExclude == 0 && geometry[sI] != 0) //we exclude the geometry, so return if we're inside of it
		return;

	
	float Bsat = MU0 * amul(Ms_, Ms_mul, sI);
    float v = amul(vol, 1.0f, sI);
	if(useBSat == 0) 
	{
		Bsat = 1.0f; //just multiply by 1, not by bsat (used with eddy currents)
	} 
	else if(useBSat == 2) 
	{
		Bsat = 1.0f;
		v = 1.0f; //Do in whole simulation volume, not just inside geometry
	}
	dst[index(ix, iy, iz, Dx, Dy, Dz)] = Bsat * v * src[sI];
}


#include "amul.h"
// absolute values of a field

extern "C" __global__ void
DissipatedPower(float* __restrict__ powers, float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz, float* __restrict__ Mx, float* __restrict__ My, float* __restrict__ Mz, float* __restrict__ alphas, float alpha, float* __restrict__ msats, float msat, float volume, float gamma, int Nx, int Ny, int Nz) {

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	if(ix >= Nx || iy >=  Ny || iz >= Nz) {
		return;
	}

	int idx = ((iz)*(Ny) + (iy))*(Nx) + (ix);

	float alphaV = amul(alphas, alpha, idx);
	float msatV = amul(msats, msat, idx);
	if(msatV == 0)
		return;

	float crossProductX = My[idx]*Bz[idx]-Mz[idx]*By[idx];
	float crossProductY = Mx[idx]*Bz[idx]-Mz[idx]*Bx[idx];
	float crossProductZ = Mx[idx]*By[idx]-My[idx]*Bx[idx];
	float preFactor = msatV*volume*gamma*alphaV/((1+alphaV*alphaV));
	
	powers[idx] = preFactor*(crossProductX*crossProductX+crossProductY*crossProductY+crossProductZ*crossProductZ);	
}


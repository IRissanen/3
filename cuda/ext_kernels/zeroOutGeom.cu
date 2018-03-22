// Make electric field zero outside the geometries

extern "C" __global__ void
zeroOutsideGeometry(float* __restrict__ EFieldX, float* __restrict__ EFieldY, float* __restrict__ EFieldZ, float* baseGeometry, float* sliderGeometry, int Nx, int Ny, int Nz) {

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	if(ix >= Nx || iy >=  Ny || iz >= Nz) {
		return;
	}

	int idx = ((iz)*(Ny) + (iy))*(Nx) + (ix);

	if(sliderGeometry[idx] != 2 && baseGeometry[idx] != 2)
	{
		EFieldX[idx] = 0;
		EFieldY[idx] = 0;
		EFieldZ[idx] = 0;
		return;
	}
}


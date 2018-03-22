// lengths of vectors in a vector field, returned in the X component

extern "C" __global__ void
Lens(float* __restrict__ FieldX, float* __restrict__ FieldY, float* __restrict__ FieldZ, int Nx, int Ny, int Nz) {

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	if(ix >= Nx || iy >=  Ny || iz >= Nz) {
		return;
	}

	int idx = ((iz)*(Ny) + (iy))*(Nx) + (ix);

	FieldX[idx] = sqrt(FieldX[idx]*FieldX[idx]+FieldY[idx]*FieldY[idx]+FieldZ[idx]*FieldZ[idx]);

}


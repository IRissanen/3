// relative differences of elements of a field, overwrites field 1

extern "C" __global__ void
RelativeDiff(float* __restrict__ Field1, float* __restrict__ Field2, int Nx, int Ny, int Nz) {

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	if(ix >= Nx || iy >=  Ny || iz >= Nz) {
		return;
	}

	int idx = ((iz)*(Ny) + (iy))*(Nx) + (ix);

	Field1[idx] = 1.0-Field2[idx]/Field1[idx];

	}


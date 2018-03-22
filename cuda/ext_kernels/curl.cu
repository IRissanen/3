// divergence of a field

extern "C" __global__ void
CurlCalc(float* __restrict__ FieldX, float* __restrict__ FieldY, float* __restrict__ FieldZ, float* __restrict__ curlX, float* __restrict__ curlY, float* __restrict__ curlZ, int Nx, int Ny, int Nz, float* includeGeometry, float cellsize) {

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	if(ix >= Nx || iy >=  Ny || iz >= Nz) {
		return;
	}


	/*if(ix >= Nx-1 || iy >=  Ny-1 || iz >= Nz-1 || ix <= 0 || iy <= 0 || iz <= 0) {
		return;
	}*/

	int idx = ((iz)*(Ny) + (iy))*(Nx) + (ix);

	int pix = ((iz)*(Ny) + (iy))*(Nx) + ((ix-1+Nx)%Nx);
	int nix = ((iz)*(Ny) + (iy))*(Nx) + ((ix+1)%Nx);
	int piy = ((iz)*(Ny) + ((iy-1+Ny)%Ny))*(Nx) + (ix);
	int niy = ((iz)*(Ny) + ((iy+1)%Ny))*(Nx) + (ix);
	int piz = (((iz-1+Nz)%Nz)*(Ny) + (iy))*(Nx) + (ix);
	int niz = (((iz+1)%Nz)*(Ny) + (iy))*(Nx) + (ix);

	if(includeGeometry[idx] == 0 || includeGeometry[pix] == 0 || includeGeometry[nix] == 0 || includeGeometry[piy] == 0 || includeGeometry[niy] == 0 || includeGeometry[piz] == 0 || includeGeometry[niz] == 0)
	{
		return;
	}

	float pxy = FieldX[piy];
	float nxy = FieldX[niy];

	float pxz = FieldX[piz];
	float nxz = FieldX[niz];

	float pyx = FieldY[pix];
	float nyx = FieldY[nix];

	float pyz = FieldY[piz];
	float nyz = FieldY[niz];

	float pzx = FieldZ[pix];
	float nzx = FieldZ[nix];

	float pzy = FieldZ[piy];
	float nzy = FieldZ[niy];


	curlX[idx] = (nzy-pzy)/(2.0*cellsize)-(nyz-pyz)/(2.0*cellsize);
	curlY[idx] = -((nzx-pzx)/(2.0*cellsize)-(nxz-pxz)/(2.0*cellsize));
	curlZ[idx] = (nyx-pyx)/(2.0*cellsize)-(nxy-pxy)/(2.0*cellsize);

	}


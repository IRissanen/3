// gradient of the electric potential field

extern "C" __global__ void
eddyGradientMove(float* __restrict__  potentialfield, float* __restrict__ EFieldX, float* __restrict__ EFieldY, float* __restrict__ EFieldZ, float* __restrict__ solFieldX, float* __restrict__ solFieldY, float* __restrict__ solFieldZ, int Nx, int Ny, int Nz, float* sliderGeometry, float* baseGeometry, float cellsize) {

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	if(ix >= Nx || iy >=  Ny || iz >= Nz) {
		return;
	}

	int idx = ((iz)*(Ny) + (iy))*(Nx) + (ix);


	if(sliderGeometry[idx] != 2 && baseGeometry[idx] != 2)
	{
		return;
	}
		


	int pix = ((iz)*(Ny) + (iy))*(Nx) + ((ix-1+Nx)%Nx);
	int nix = ((iz)*(Ny) + (iy))*(Nx) + ((ix+1)%Nx);
	int piy = ((iz)*(Ny) + ((iy-1+Ny)%Ny))*(Nx) + (ix);
	int niy = ((iz)*(Ny) + ((iy+1)%Ny))*(Nx) + (ix);
	int piz = (((iz-1+Nz)%Nz)*(Ny) + (iy))*(Nx) + (ix);
	int niz = (((iz+1)%Nz)*(Ny) + (iy))*(Nx) + (ix);
	float px = potentialfield[pix];
	float nx = potentialfield[nix];
	float py = potentialfield[piy];
	float ny = potentialfield[niy];
	float pz = potentialfield[piz];
	float nz = potentialfield[niz];

	//normal directions at the boundaries
 	//for this to work, the magnets MUST be more than 1 cell thick in each direction!
 	//If they're not, then the surfaces technically has 2 normals, and these cancel

	int norX = (sliderGeometry[idx] == 2 || baseGeometry[idx] == 2)*(-(sliderGeometry[pix] == 1 || baseGeometry[pix] == 1) + (sliderGeometry[nix] == 1 || baseGeometry[nix] == 1)); 
	int norY = (sliderGeometry[idx] == 2 || baseGeometry[idx] == 2)*(-(sliderGeometry[piy] == 1 || baseGeometry[piy] == 1) + (sliderGeometry[niy] == 1 || baseGeometry[niy] == 1));
	int norZ = (sliderGeometry[idx] == 2 || baseGeometry[idx] == 2)*(-(sliderGeometry[piz] == 1 || baseGeometry[piz] == 1) + (sliderGeometry[niz] == 1 || baseGeometry[niz] == 1)); 

	float negativeNormalProjection = -(solFieldX[idx]*norX+solFieldY[idx]*norY+solFieldZ[idx]*norZ)/(norX*norX+norY*norY+norZ*norZ);

	if(norX != 0) 
		EFieldX[idx] = negativeNormalProjection*norX;
	else
		EFieldX[idx] = -(nx - px)/(2.0*cellsize);

	if(norY != 0) 
		EFieldY[idx] = negativeNormalProjection*norY;
	else
		EFieldY[idx] = -(ny - py)/(2.0*cellsize);

	if(norZ != 0) 
		EFieldZ[idx] = negativeNormalProjection*norZ;
	else 
		EFieldZ[idx] = -(nz - pz)/(2.0*cellsize);

}


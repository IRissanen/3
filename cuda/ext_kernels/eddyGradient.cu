// gradient of the electric potential field
//the variable "m" is to help with moving geometries, for which there are values of 2 (geometry), 1 (the nearest neighbor cells next to geometry) and 0 instead of just 1 and 0

extern "C" __global__ void
eddyGradient(float* __restrict__  potentialfield, float* __restrict__ EFieldX, float* __restrict__ EFieldY, float* __restrict__ EFieldZ, float* __restrict__ solFieldX, float* __restrict__ solFieldY, float* __restrict__ solFieldZ, int pbcX, int pbcY, int pbcZ,  int Nx, int Ny, int Nz, float* geometry, int m, float cellsize) {

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	if(ix >= Nx || iy >=  Ny || iz >= Nz) {
		return;
	}

	int idx = ((iz)*(Ny) + (iy))*(Nx) + (ix);		

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
 	//for this to work, the magnet MUST be more than 1 cell thick in each direction!
 	//If it's not, then the surfaces technically has 2 normals, and these cancel

	int norX = (geometry[idx] == 1+m)*(-(geometry[pix] == 0+m || (pbcX == 0 && ix == 0)) + (geometry[nix] == 0+m || (pbcX == 0 && ix == Nx-1))); 
	int norY = (geometry[idx] == 1+m)*(-(geometry[piy] == 0+m || (pbcY == 0 && iy == 0)) + (geometry[niy] == 0+m || (pbcY == 0 && iy == Ny-1))); 
	int norZ = (geometry[idx] == 1+m)*(-(geometry[piz] == 0+m || (pbcZ == 0 && iz == 0)) + (geometry[niz] == 0+m || (pbcZ == 0 && iz == Nz-1))); 

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


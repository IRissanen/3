//Solve the Laplace equation via Successive-Over-Relaxation on the GPU
//the variable "m" is to help with moving geometries, for which there are values of 2 (geometry), 1 (the nearest neighbor cells next to geometry) and 0 instead of just 1 and 0

extern "C" {

__global__ void eddyLaplace(float* __restrict__  field, float* __restrict__ solFieldX, float* __restrict__ solFieldY, float* __restrict__ solFieldZ, int pbcX, int pbcY, int pbcZ, int Nx, int Ny, int Nz, int even, float* __restrict__ geometry, int m, float cellsize) {

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	if(ix >= Nx || iy >=  Ny || iz >= Nz) {
		return;
	}

	int idx = ((iz)*(Ny) + (iy))*(Nx) + (ix);

	if(even == 1 && !((ix+iy+iz)%2))
		return;
	else if (even == 0 && (ix+iy+iz)%2)
		return;

	//Here we wrap around indices regardless, simulation boundaries set to 0 if not periodic
	int pix = ((iz)*(Ny) + (iy))*(Nx) + ((ix-1+Nx)%Nx);
	int nix = ((iz)*(Ny) + (iy))*(Nx) + ((ix+1)%Nx);
	int piy = ((iz)*(Ny) + ((iy-1+Ny)%Ny))*(Nx) + (ix);
	int niy = ((iz)*(Ny) + ((iy+1)%Ny))*(Nx) + (ix);
	int piz = (((iz-1+Nz)%Nz)*(Ny) + (iy))*(Nx) + (ix);
	int niz = (((iz+1)%Nz)*(Ny) + (iy))*(Nx) + (ix);
	float px = (pbcX > 0 || ix > 0)*field[pix];
	float nx = (pbcX > 0 || ix < Nx-1)*field[nix];
	float py = (pbcY > 0 || iy > 0)*field[piy];
	float ny = (pbcY > 0 || iy < Ny-1)*field[niy];
	float pz = (pbcZ > 0 || iz > 0)*field[piz];
	float nz = (pbcZ > 0 || iz < Nz-1)*field[niz];


	//Normal component of E_solenoidal, with some boolean calculations

	int norX = (geometry[idx] == 1+m)*(-(geometry[pix] == 0+m || (pbcX == 0 && ix == 0)) + (geometry[nix] == 0+m || (pbcX == 0 && ix == Nx-1))); 
	int norY = (geometry[idx] == 1+m)*(-(geometry[piy] == 0+m || (pbcY == 0 && iy == 0)) + (geometry[niy] == 0+m || (pbcY == 0 && iy == Ny-1))); 
	int norZ = (geometry[idx] == 1+m)*(-(geometry[piz] == 0+m || (pbcZ == 0 && iz == 0)) + (geometry[niz] == 0+m || (pbcZ == 0 && iz == Nz-1))); 

	float normalProjection = (solFieldX[idx]*norX+solFieldY[idx]*norY+solFieldZ[idx]*norZ)/(norX*norX+norY*norY+norZ*norZ);
	
	//Normal derivative finite differences, taking into account periodic boundaries and neumann boundary conditions at geometry.


	if(norX == -1)
		px = 2.0*normalProjection*cellsize+nx;
	else if(norX == 1)
		nx = 2.0*normalProjection*cellsize+px;

	if(norY == -1)
		py = 2.0*normalProjection*cellsize+ny;
	else if(norY == 1)
		ny = 2.0*normalProjection*cellsize+py;

	if(norZ == -1)
		pz = 2.0*normalProjection*cellsize+nz;
	else if(norZ == 1)
		nz = 2.0*normalProjection*cellsize+pz;

	field[idx] = (-0.4)*field[idx]+(1.4/6.0)*(px+nx+py+ny+pz+nz);
}

}

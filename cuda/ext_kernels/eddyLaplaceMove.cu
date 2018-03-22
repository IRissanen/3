//Solve the Laplace equation via Successive Over Relaxation on the GPU

extern "C" {

__global__ void eddyLaplaceMove(float* __restrict__  field, float* __restrict__ solFieldX, float* __restrict__ solFieldY, float* __restrict__ solFieldZ, int Nx, int Ny, int Nz, float* sliderGeometry, float* baseGeometry, int even, float cellsize) {

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

	int pix = ((iz)*(Ny) + (iy))*(Nx) + ((ix-1+Nx)%Nx);
	int nix = ((iz)*(Ny) + (iy))*(Nx) + ((ix+1)%Nx);
	int piy = ((iz)*(Ny) + ((iy-1+Ny)%Ny))*(Nx) + (ix);
	int niy = ((iz)*(Ny) + ((iy+1)%Ny))*(Nx) + (ix);
	int piz = (((iz-1+Nz)%Nz)*(Ny) + (iy))*(Nx) + (ix);
	int niz = (((iz+1)%Nz)*(Ny) + (iy))*(Nx) + (ix);
	float px = field[pix];
	float nx = field[nix];
	float py = field[piy];
	float ny = field[niy];
	float pz = field[piz];
	float nz = field[niz];


	//Normal component of E_solenoidal

	int norX = (sliderGeometry[idx] == 2 || baseGeometry[idx] == 2)*(-(sliderGeometry[pix] == 1 || baseGeometry[pix] == 1) + (sliderGeometry[nix] == 1 || baseGeometry[nix] == 1)); 
	int norY = (sliderGeometry[idx] == 2 || baseGeometry[idx] == 2)*(-(sliderGeometry[piy] == 1 || baseGeometry[piy] == 1) + (sliderGeometry[niy] == 1 || baseGeometry[niy] == 1));
	int norZ = (sliderGeometry[idx] == 2 || baseGeometry[idx] == 2)*(-(sliderGeometry[piz] == 1 || baseGeometry[piz] == 1) + (sliderGeometry[niz] == 1 || baseGeometry[niz] == 1)); 

	float normalProjection = (solFieldX[idx]*norX+solFieldY[idx]*norY+solFieldZ[idx]*norZ)/(norX*norX+norY*norY+norZ*norZ);
	
	//Normal derivative finite differences, taking into account periodic boundaries

	if(sliderGeometry[idx] == 2)
	{
		if(sliderGeometry[pix] == 1)
			px = 2.0*normalProjection*cellsize+nx;
		else if(sliderGeometry[nix] == 1)
			nx = 2.0*normalProjection*cellsize+px;

		if(sliderGeometry[piy] == 1)
			py = 2.0*normalProjection*cellsize+ny;
		else if(sliderGeometry[niy] == 1)
			ny = 2.0*normalProjection*cellsize+py;

		if(sliderGeometry[piz] == 1)
			pz = 2.0*normalProjection*cellsize+nz;
		else if(sliderGeometry[niz] == 1)
			nz = 2.0*normalProjection*cellsize+pz;
	}
	else if(baseGeometry[idx] == 2)
	{
		if(baseGeometry[pix] == 1)
			px = 2.0*normalProjection*cellsize+nx;
		else if(baseGeometry[nix] == 1)
			nx = 2.0*normalProjection*cellsize+px;

		if(baseGeometry[piy] == 1)
			py = 2.0*normalProjection*cellsize+ny;
		else if(baseGeometry[niy] == 1)
			ny = 2.0*normalProjection*cellsize+py;

		if(baseGeometry[piz] == 1)
			pz = 2.0*normalProjection*cellsize+nz;
		else if(baseGeometry[niz] == 1)
			nz = 2.0*normalProjection*cellsize+pz;
	}

	field[idx] = (-0.4)*field[idx]+(1.4/6.0)*(px+nx+py+ny+pz+nz);
}

}
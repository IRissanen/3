// four-point gradient of a scalar field (src), with the possibility to do it only in certain geometry or exclude certain geometry

extern "C" __global__ void
Gradient(float* __restrict__ dstX, float* __restrict__ dstY, float* __restrict__ dstZ, float* __restrict__ src, int Nx, int Ny, int Nz, int Sx, int Sy, int Sz, float* geometry, int includeOrExclude, float cellsize, int cellshift) {

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	if(ix >= Nx || iy >= Ny || iz >= Nz)
	{
		return;
	}

	int idx = ((iz)*(Ny) + (iy))*(Nx) + (ix);

	ix += cellshift; //shift indices due to padding of scalar potential from the sides
	iy += cellshift;
	iz += cellshift;

	if(includeOrExclude == 1 && geometry[idx] == 0)  //When including only the geometry, we don't take gradient outside of it.
	{
		return;
	}
	else if(includeOrExclude == 0 && geometry[idx] != 0) //we exclude the geometry, so don't take gradient if we're inside of it
	{
		return;
	}
		
	int ixp = (iz*Sy + iy)*Sx + ix-1;
	int iyp = (iz*Sy + iy-1)*Sx + ix;
	int izp = ((iz-1)*Sy + iy)*Sx + ix;
	int ixn = (iz*Sy + iy)*Sx + ix+1;
	int iyn = (iz*Sy + iy+1)*Sx + ix;
	int izn = ((iz+1)*Sy + iy)*Sx + ix;
		

	int ixpp = (iz*Sy + iy)*Sx + ix-2;
	int iypp = (iz*Sy + iy-2)*Sx + ix;
	int izpp = ((iz-2)*Sy + iy)*Sx + ix;
	int ixnn = (iz*Sy + iy)*Sx + ix+2;
	int iynn = (iz*Sy + iy+2)*Sx + ix;
	int iznn = ((iz+2)*Sy + iy)*Sx + ix;
			
	float xn = src[ixn];
	float xp = src[ixp];
	float yn = src[iyn];
	float yp = src[iyp];
	float zn = src[izn];
	float zp = src[izp];

	float xnn = src[ixnn];
	float xpp = src[ixpp];
	float ynn = src[iynn];
	float ypp = src[iypp];
	float znn = src[iznn];
	float zpp = src[izpp];
	
	dstX[idx] += (xnn-xpp)/(12.0*cellsize)+8*(xp-xn)/(12.0*cellsize); //4-point finite difference
	dstY[idx] += (ynn-ypp)/(12.0*cellsize)+8*(yp-yn)/(12.0*cellsize);
	dstZ[idx] += (znn-zpp)/(12.0*cellsize)+8*(zp-zn)/(12.0*cellsize);

}
#include <stdio.h>
//#include <cutil.h>
#include "memcpy_c.cu"
#include "cubicPrefilter3D_c.cu"
#include "cubicTex3D_c.cu"

extern "C" {

#include "interpolationKernel.h"

texture<float, 3, cudaReadModeElementType> tex;
cudaArray *coeffArray = 0;

__global__ void InterpolateFieldKernel(float* newField, int Nx, int Ny, int Nz, float mx, float my, float mz, int cellshift, float* geometry, int includeOrExclude) //0 for excluding the geometry, 1 for including only the geometry, 2 for including everything.
{
	//interpolate a single vector component
	
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	if(ix >= Nx || iy >= Ny || iz >= Nz)
	{
		return;
	}

	int idx = ((iz)*(Ny) + (iy))*(Nx) + (ix);

	if(includeOrExclude == 1 && geometry[idx] == 0) //When including only the geometry, we don't interpolate outside of it.
	{
		mx = 0.0; my = 0.0; mz = 0.0; //local copies, so this doesn't change the actual values.
	}
	else if(includeOrExclude == 0 && geometry[idx] != 0) //we exclude the geometry, so don't interpolate movement if we're inside of it
	{
		mx = 0.0; my = 0.0; mz = 0.0;
	}

	float u = ((float)(ix + cellshift) + 0.5 + mx); //mx etc. = normalized movement between cells 
	float v = ((float)(iy + cellshift) + 0.5 + my);
	float w = ((float)(iz + cellshift) + 0.5 + mz);
	float3 coords = make_float3(u, v, w);
	// read new component value from the 3D texture at interpolated points.
	newField[idx] = cubicTex3DSimple(tex, coords);
}

void InterpolateField(float* newComp, int width, int height, int depth, float mx, float my, float mz, int cellshift, float* geometry, int includeOrExclude)
{
	dim3 dimBlock(32, 32, 1);
	dim3 dimGrid((width / dimBlock.x)+1, (height / dimBlock.y)+1, depth+1);
	InterpolateFieldKernel<<<dimGrid, dimBlock>>>(newComp, width, height, depth, mx, my, mz, cellshift, geometry, includeOrExclude);
}

void NoPrecondition(float* pot, int Nx, int Ny, int Nz)
{
	//precondition data and make a texture out of it.
	const cudaExtent volumeExtent = make_cudaExtent(Nx, Ny, Nz);
	cudaPitchedPtr bsplineCoeffs = make_cudaPitchedPtr((void*)pot, volumeExtent.width*sizeof(float), volumeExtent.width, volumeExtent.height);

	//CubicBSplinePrefilter3D((float*)bsplineCoeffs.ptr, (uint)bsplineCoeffs.pitch, Nx, Ny, Nz);
	
	CreateTextureFromVolumeWithPointer(&tex, &coeffArray, bsplineCoeffs, volumeExtent, true);
	tex.filterMode = cudaFilterModeLinear;
	tex.normalized = false; 
    tex.addressMode[0] = cudaAddressModeMirror;
	//CUDA_SAFE_CALL(cudaFree(bsplineCoeffs.ptr)); 
}

void Precondition(float* pot, int Nx, int Ny, int Nz)
{
	//precondition data and make a texture out of it.
	const cudaExtent volumeExtent = make_cudaExtent(Nx, Ny, Nz);
	cudaPitchedPtr bsplineCoeffs = make_cudaPitchedPtr((void*)pot, volumeExtent.width*sizeof(float), volumeExtent.width, volumeExtent.height);

	CubicBSplinePrefilter3D((float*)bsplineCoeffs.ptr, (uint)bsplineCoeffs.pitch, Nx, Ny, Nz);
	
	CreateTextureFromVolumeWithPointer(&tex, &coeffArray, bsplineCoeffs, volumeExtent, true);
	tex.filterMode = cudaFilterModeLinear;
	tex.normalized = false; 
    tex.addressMode[0] = cudaAddressModeMirror;
	//CUDA_SAFE_CALL(cudaFree(bsplineCoeffs.ptr)); 
}

}
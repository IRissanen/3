/*--------------------------------------------------------------------------*\
Copyright (c) 2008-2010, Danny Ruijters. All rights reserved.
http://www.dannyruijters.nl/cubicinterpolation/
This file is part of CUDA Cubic B-Spline Interpolation (CI).

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
*  Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
*  Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
*  Neither the name of the copyright holders nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are
those of the authors and should not be interpreted as representing official
policies, either expressed or implied.

When using this code in a scientific project, please cite one or all of the
following papers:
*  Daniel Ruijters and Philippe Th�venaz,
   GPU Prefilter for Accurate Cubic B-Spline Interpolation, 
   The Computer Journal, vol. 55, no. 1, pp. 15-20, January 2012.
   http://dannyruijters.nl/docs/cudaPrefilter3.pdf
*  Daniel Ruijters, Bart M. ter Haar Romeny, and Paul Suetens,
   Efficient GPU-Based Texture Interpolation using Uniform B-Splines,
   Journal of Graphics Tools, vol. 13, no. 4, pp. 61-69, 2008.
\*--------------------------------------------------------------------------*/

#ifndef _MEMCPY_CUDA_H_
#define _MEMCPY_CUDA_H_

extern "C" {

#include <stdio.h>
//#include <cutil.h>
#include "internal/math_func_c.cu"


cudaPitchedPtr CopyVolumeHostToDevice(const float* host, uint width, uint height, uint depth)
{
	cudaPitchedPtr device = {0};
	const cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth);
	cudaMalloc3D(&device, extent);
	cudaMemcpy3DParms p = {0};
	p.srcPtr = make_cudaPitchedPtr((void*)host, width * sizeof(float), width, height);
	p.dstPtr = device;
	p.extent = extent;
	p.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&p);
	return device;
}

//! Copy a voxel volume from a pitched pointer to a texture
//! @param tex      [output]  pointer to the texture
//! @param texArray [output]  pointer to the texArray
//! @param volume   [input]   pointer to the the pitched voxel volume
//! @param extent   [input]   size (width, height, depth) of the voxel volume
//! @param onDevice [input]   boolean to indicate whether the voxel volume resides in GPU (true) or CPU (false) memory
//! @note When the texArray is not yet allocated, this function will allocate it
void CreateTextureFromVolumeWithPointer(texture<float, cudaTextureType3D, cudaReadModeElementType>* tex, cudaArray** texArray, const cudaPitchedPtr volume, cudaExtent extent, bool onDevice)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	if (*texArray == 0) cudaMalloc3DArray(texArray, &channelDesc, extent);
	// copy data to 3D array
	cudaMemcpy3DParms p = {0};
	p.extent   = extent;
	p.srcPtr   = volume;
	p.dstArray = *texArray;
	p.kind     = onDevice ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
	cudaMemcpy3D(&p);
	// bind array to 3D texture
	cudaBindTextureToArray(*tex, *texArray, channelDesc);
	tex->normalized = false;  //access with absolute texture coordinates
	tex->filterMode = cudaFilterModeLinear;
}

//! Copy a voxel volume from continuous memory to a texture
//! @param tex      [output]  pointer to the texture
//! @param texArray [output]  pointer to the texArray
//! @param volume   [input]   pointer to the continuous memory with the voxel
//! @param extent   [input]   size (width, height, depth) of the voxel volume
//! @param onDevice [input]   boolean to indicate whether the voxel volume resides in GPU (true) or CPU (false) memory
//! @note When the texArray is not yet allocated, this function will allocate it

void CreateTextureFromVolume(texture<float, cudaTextureType3D, cudaReadModeElementType>* tex, cudaArray** texArray, const float* volume, cudaExtent extent, bool onDevice)
{
	cudaPitchedPtr ptr = make_cudaPitchedPtr((void*)volume, extent.width*sizeof(float), extent.width, extent.height);
	CreateTextureFromVolumeWithPointer(tex, texArray, ptr, extent, onDevice);
}

}

#endif  //_MEMCPY_CUDA_H_
package cuda

// Additional kernel multiplications and other functions using cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"github.com/mumax/3/cuda/ext_kernels"
	"github.com/mumax/3/cuda/cu"
)


// kernel multiplication for calculating eddy currents, using the scalar potential kernel
func Ext_move_eddykernMulRSymm3D_async(fftM [3]*data.Slice, Kxx, Kyy, Kzz *data.Slice, Nx, Ny, Nz int) {
	util.Argument(fftM[X].NComp() == 1 && Kxx.NComp() == 1)

	cfg := ext_kernels.Make3DConf([3]int{Nx, Ny, Nz})
	ext_kernels.K_ext_eddykernmulRSymm3D_async(fftM[X].DevPtr(0), fftM[Y].DevPtr(0), fftM[Z].DevPtr(0),
		Kxx.DevPtr(0), Kyy.DevPtr(0), Kzz.DevPtr(0),
		Nx, Ny, Nz, cfg)
}

// Successive-Over-Relaxation kernel for solving Laplacian in eddy current calculation
func Ext_EddyLaplace_async(dstField, solField *data.Slice, pbc, size [3]int, even int, geometry *data.Slice, moving int, cellsize float32) {	
	cfg := ext_kernels.Make3DConf([3]int{size[X], size[Y], size[Z]})
	ext_kernels.K_ext_eddyLaplace_async(dstField.DevPtr(0), solField.Comp(X).DevPtr(0), solField.Comp(Y).DevPtr(0), solField.Comp(Z).DevPtr(0), pbc[X], pbc[Y], pbc[Z], size[X], size[Y], size[Z], even, geometry.DevPtr(0), moving, cellsize, cfg)
}

//Calculation of pointwise divergence (central finite difference)
func Ext_move_DivergenceCalc_async(Field, excludeGeometry, divergence *data.Slice, Nx, Ny, Nz int, cellsize float32) {
	
	cfg := ext_kernels.Make3DConf([3]int{Nx, Ny, Nz})
	ext_kernels.K_ext_DivergenceCalc_async(Field.Comp(X).DevPtr(0), Field.Comp(Y).DevPtr(0), Field.Comp(Z).DevPtr(0), divergence.DevPtr(0), Nx, Ny, Nz, excludeGeometry.DevPtr(0), cellsize, cfg)
}

//Calculation of pointwise curls (central finite difference)
func Ext_move_CurlCalc_async(Field, excludeGeometry, curl *data.Slice, Nx, Ny, Nz int, cellsize float32) {
	
	cfg := ext_kernels.Make3DConf([3]int{Nx, Ny, Nz})
	ext_kernels.K_ext_CurlCalc_async(Field.Comp(X).DevPtr(0), Field.Comp(Y).DevPtr(0), Field.Comp(Z).DevPtr(0), curl.Comp(X).DevPtr(0), curl.Comp(Y).DevPtr(0), curl.Comp(Z).DevPtr(0), Nx, Ny, Nz, excludeGeometry.DevPtr(0), cellsize, cfg)
}

//Gradient calculation taking the Neumann boundary conditions into account when determining eddy currents
func Ext_EddyGradient_async(potentialField, irrotField, solField *data.Slice, pbc, size [3]int, geometry *data.Slice, moving int, cellsize float32) {
	
	cfg := ext_kernels.Make3DConf([3]int{size[X], size[Y], size[Z]})
	ext_kernels.K_ext_eddyGradient_async(potentialField.DevPtr(0), irrotField.Comp(X).DevPtr(0), irrotField.Comp(Y).DevPtr(0), irrotField.Comp(Z).DevPtr(0), solField.Comp(X).DevPtr(0), solField.Comp(Y).DevPtr(0), solField.Comp(Z).DevPtr(0), pbc[X], pbc[Y], pbc[Z], size[X], size[Y], size[Z], geometry.DevPtr(0), moving, cellsize, cfg)
}

//Just normal gradient, with possibility of including or excluding geometry
func Ext_move_Gradient_async(dst, src *data.Slice, dstSize, srcSize [3]int, geometry *data.Slice, includeOrExclude int, cellsize float32, cellshift int) {
	
	cfg := ext_kernels.Make3DConf([3]int{dstSize[0], dstSize[1], dstSize[2]})
	ext_kernels.K_ext_Gradient_async(dst.Comp(X).DevPtr(0), dst.Comp(Y).DevPtr(0), dst.Comp(Z).DevPtr(0), src.DevPtr(0), dstSize[0], dstSize[1], dstSize[2],  srcSize[0], srcSize[1], srcSize[2], geometry.DevPtr(0), includeOrExclude, cellsize, cellshift, cfg)
}

// kernel multiplication for 3D potential field convolution, exploiting full kernel symmetry.
func Ext_move_spkernMulRSymm3D_async(fftM [3]*data.Slice, Kxx, Kyy, Kzz *data.Slice, Nx, Ny, Nz int) {
	util.Argument(fftM[X].NComp() == 1 && Kxx.NComp() == 1)

	cfg := ext_kernels.Make3DConf([3]int{Nx, Ny, Nz})
	ext_kernels.K_ext_spkernmulRSymm3D_async(fftM[X].DevPtr(0), fftM[Y].DevPtr(0), fftM[Z].DevPtr(0),
		Kxx.DevPtr(0), Kyy.DevPtr(0), Kzz.DevPtr(0),
		Nx, Ny, Nz, cfg)
}

//Taking absolute values of a field, storing in place
func Ext_move_CudaAbs(dst *data.Slice, dstsize [3]int) {

	util.Argument(dst.NComp() == 1)
	util.Argument(dst.Len() == prod(dstsize))

	cfg := ext_kernels.Make3DConf(dstsize)

	ext_kernels.K_ext_AbsField_async(dst.DevPtr(0), dstsize[X], dstsize[Y], dstsize[Z], cfg)
}

//Taking length of vectors in a field, storing in the X-component.
func Ext_move_CudaLens(dst *data.Slice, dstsize [3]int) {

	util.Argument(dst.NComp() == 3)
	util.Argument(dst.Len() == prod(dstsize))

	cfg := ext_kernels.Make3DConf(dstsize)

	ext_kernels.K_ext_Lens_async(dst.Comp(0).DevPtr(0), dst.Comp(1).DevPtr(0), dst.Comp(2).DevPtr(0), dstsize[X], dstsize[Y], dstsize[Z], cfg)
}

//The elementwise relative difference between two fields, with the dst being the divisor
func Ext_move_RelativeDiff(dst, src *data.Slice, dstsize [3]int) {

	util.Argument(dst.NComp() == 1)
	util.Argument(dst.Len() == prod(dstsize))

	cfg := ext_kernels.Make3DConf(dstsize)

	ext_kernels.K_ext_RelativeDiff_async(dst.DevPtr(0), src.DevPtr(0), dstsize[X], dstsize[Y], dstsize[Z], cfg)
}

//Cuda calculation of dissipated power due to vector precession
func Ext_move_dissipatedPowers(dst, Mag, B *data.Slice, alphas, msats MSlice, volume, gamma float64, dstsize [3]int) {

	util.Argument(dst.NComp() == 1)
	util.Argument(dst.Len() == prod(dstsize))

	cfg := ext_kernels.Make3DConf(dstsize)

	ext_kernels.K_ext_DissipatedPower_async(dst.DevPtr(0), B.DevPtr(0), B.DevPtr(1), B.DevPtr(2), Mag.DevPtr(0), Mag.DevPtr(1), Mag.DevPtr(2), alphas.DevPtr(0), alphas.Mul(0), msats.DevPtr(0), msats.Mul(0), float32(volume), float32(gamma), dstsize[X], dstsize[Y], dstsize[Z], cfg)
}

//Elementwise absolute differences between two fields.
func Ext_move_AbsDiff(dst, src *data.Slice, dstsize [3]int) {

	util.Argument(dst.NComp() == 1)
	util.Argument(dst.Len() == prod(dstsize))

	cfg := ext_kernels.Make3DConf(dstsize)

	ext_kernels.K_ext_AbsDiff_async(dst.DevPtr(0), src.DevPtr(0), dstsize[X], dstsize[Y], dstsize[Z], cfg)
}

//Copying and padding, with the possibility of including or excluding certain geometry and not using Bsat.
//useBsat: 0 = Use 1.0 instead of Bsat, 1 = Use Bsat normally, 2 = Use 1.0 in place of Bsat and do it in whole simulation domain (eddy currents utilize this).
func Ext_move_copyPadMulGeom(dst, src, vol *data.Slice, dstsize, srcsize [3]int, geometry *data.Slice, useBSat, includeOrExclude int, Msat MSlice) {
	util.Argument(dst.NComp() == 1 && src.NComp() == 1)
	util.Assert(dst.Len() == prod(dstsize) && src.Len() == prod(srcsize))

	cfg := ext_kernels.Make3DConf(srcsize)

	ext_kernels.K_ext_copypadmulGeom_async(dst.DevPtr(0), dstsize[X], dstsize[Y], dstsize[Z],
		src.DevPtr(0), vol.DevPtr(0), srcsize[X], srcsize[Y], srcsize[Z],
		Msat.DevPtr(0), Msat.Mul(0), geometry.DevPtr(0), useBSat, includeOrExclude, cfg)
}

//CopyUnpadding with the possibility of including or excluding certain geometry, and copying boundary values over periodic boundaries
func Ext_move_copyUnpadGeom(dst, src *data.Slice, dstsize, srcsize [3]int, sShift, periodic int, geometry *data.Slice, includeOrExclude int) {
	util.Argument(dst.NComp() == 1 && src.NComp() == 1)
	util.Assert(dst.Len() == prod(dstsize) && src.Len() == prod(srcsize))

	cfg := ext_kernels.Make3DConf(dstsize)

	ext_kernels.K_ext_copyunpadGeom_async(dst.DevPtr(0), dstsize[X], dstsize[Y], dstsize[Z],
		src.DevPtr(0), srcsize[X], srcsize[Y], srcsize[Z], sShift, periodic, geometry.DevPtr(0), includeOrExclude, cfg)
}

// Shift functions for specific zones
func Ext_move_ShiftXZone(dst, src *data.Slice, shiftVec [3]int, clampL, clampR float32, shiftGeom *data.Slice) {
	
	util.Argument(dst.NComp() == 1 && src.NComp() == 1)
	util.Assert(dst.Len() == src.Len())
	N := dst.Size()
	cfg := ext_kernels.Make3DConf(N)
	ext_kernels.K_ext_shiftxzone_async(dst.DevPtr(0), src.DevPtr(0), N[X], N[Y], N[Z], shiftVec[X], shiftVec[Y], shiftVec[Z], clampL, clampR, shiftGeom.DevPtr(0), cfg)
}

func Ext_move_ShiftBytesZone(dst, src *Bytes, m *data.Mesh, shiftVec [3]int, clamp byte, shiftGeom *data.Slice) {
	N := m.Size()
	cfg := ext_kernels.Make3DConf(N)
	ext_kernels.K_ext_shiftbyteszone_async(dst.Ptr, src.Ptr, N[X], N[Y], N[Z], shiftVec[X], shiftVec[Y], shiftVec[Z], clamp, shiftGeom.DevPtr(0), cfg)
}

//Zeroing which is callable from elsewhere
func Zero1_async(dst *data.Slice) {
	cu.MemsetD32Async(cu.DevicePtr(uintptr(dst.DevPtr(0))), 0, int64(dst.Len()), stream0)
}

//componentwise multiplication by Mslice
func MSliceMul(dst, src1 *data.Slice, src2 MSlice) {
	util.Argument(dst.NComp() == 1 && src1.NComp() == 1)
	util.Assert(dst.Len() == src1.Len())
	N := dst.Size()
	cfg := ext_kernels.Make3DConf(N)
	ext_kernels.K_ext_MSliceMul_async(dst.DevPtr(0), src1.DevPtr(0), src2.DevPtr(0), src2.Mul(0), N[X], N[Y], N[Z], cfg)
}
package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/cuda/ext_move_cuinterp"
)

//Calculates the non-interpolated demagnetizing field, with possibility to exclude certain part from creating the field (still receives the field, unlike NoDemag)
func Exec3DExcludeGen(c *DemagConvolution, outp, inp, vol *data.Slice, targetGeom *data.Slice, includeOrExclude int, msat MSlice) {

	for i := 0; i < 3; i++ { // FW FFT
		fwFFT(c, i, inp, vol, targetGeom, includeOrExclude, msat)
	}

	// kern mul
	kernMulRSymm3D_async(c.fftCBuf,
		c.kern[X][X], c.kern[Y][Y], c.kern[Z][Z],
		c.kern[Y][Z], c.kern[X][Z], c.kern[X][Y],
		c.fftKernLogicSize[X], c.fftKernLogicSize[Y], c.fftKernLogicSize[Z])

	for i := 0; i < 3; i++ { // BW FFT
		bwFFT(c, i, outp, targetGeom, 2, false)
	}
}

//In this case, only the part which creates field receives it (similar to NoDemag).
func Exec3DExcludeTotal(c *DemagConvolution, outp, inp, vol *data.Slice, targetGeom *data.Slice, includeOrExclude int, Msat MSlice) {
	for i := 0; i < 3; i++ { // FW FFT
		fwFFT(c, i, inp, vol, targetGeom, includeOrExclude, Msat)
	}

	// kern mul
	kernMulRSymm3D_async(c.fftCBuf,
		c.kern[X][X], c.kern[Y][Y], c.kern[Z][Z],
		c.kern[Y][Z], c.kern[X][Z], c.kern[X][Y],
		c.fftKernLogicSize[X], c.fftKernLogicSize[Y], c.fftKernLogicSize[Z])

	for i := 0; i < 3; i++ { // BW FFT
		bwFFT(c, i, outp, targetGeom, includeOrExclude, false)
	}
}

//Calculates and interpolates the demagnetizing field, with possibility to exclude certain part from creating the field (still receives the field, unlike NoDemag)
func Exec3DInterp(c *DemagConvolution, outp, inp, vol *data.Slice, mx, my, mz, cellsize float64, targetGeom *data.Slice, includeOrExclude int, Msat MSlice) {
	for i := 0; i < 3; i++ { // FW FFT
		fwFFT(c, i, inp, vol, targetGeom, includeOrExclude, Msat)
	}
	// kern mul
	kernMulRSymm3D_async(c.fftCBuf,
		c.kern[X][X], c.kern[Y][Y], c.kern[Z][Z],
		c.kern[Y][Z], c.kern[X][Z], c.kern[X][Y],
		c.fftKernLogicSize[X], c.fftKernLogicSize[Y], c.fftKernLogicSize[Z])

	//Larger buffer so we get enough points for preconditioning and interpolation
	size := c.inputSize
	size[X] += 8
	size[Y] += 8
	size[Z] += 8
	largebuffer := Buffer(1, size)
	defer Recycle(largebuffer)
	zero1_async(largebuffer)
	for i := 0; i < 3; i++ { // BW FFT
		bwFFT(c, i, largebuffer, targetGeom, 2, true) //The result BWFFT'd into a larger buffer so that the interpolation can use more points to get more accurate results at the boundary
		ext_move_cuinterp.Precondition(largebuffer, size)
		ext_move_cuinterp.InterpolateField(outp.Comp(i), c.inputSize, float32(mx), float32(my), float32(mz), 4, targetGeom, 1-includeOrExclude)
	}
}

// forward FFT component i, with possibility to exclude or include only certain geometry.
func fwFFT(c *DemagConvolution, i int, inp, vol *data.Slice, targetGeom *data.Slice, includeOrExclude int, Msat MSlice) {
	zero1_async(c.fftRBuf[i])
	in := inp.Comp(i)
	Ext_move_copyPadMulGeom(c.fftRBuf[i], in, vol, c.realKernSize, in.Size(), targetGeom, 1, includeOrExclude, Msat)
	c.fwPlan.ExecAsync(c.fftRBuf[i], c.fftCBuf[i])
}

// backward FFT component i, with possibility to exclude or include only certain geometry.
// For interpolation, we need extra points outside the boundary of the simulation domain (largerbuffer). Either copy these from the opposite side (periodic),
// or copy them from the larger convolution, since in the non-periodic case the kernel and magnetization are padded larger anyway.
func bwFFT(c *DemagConvolution, i int, outp *data.Slice, targetGeom *data.Slice, includeOrExclude int, largerBuffer bool) {
	c.bwPlan.ExecAsync(c.fftCBuf[i], c.fftRBuf[i])
	if(largerBuffer) {
		out := outp.Comp(X)		
		Ext_move_copyUnpadGeom(out, c.fftRBuf[i], out.Size(), c.realKernSize, -4, 1, targetGeom, includeOrExclude)
	} else {
		out := outp.Comp(i)
		Ext_move_copyUnpadGeom(out, c.fftRBuf[i], out.Size(), c.realKernSize, 0, 0, targetGeom, includeOrExclude)
	}
}
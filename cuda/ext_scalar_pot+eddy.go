package cuda

import (
	"github.com/mumax/3/cuda/ext_move_cuinterp"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"math"
	"log"
)

// Stores the necessary state to perform FFT-accelerated convolution
// with scalar potential kernel (or other kernel of same symmetry).
type PotentialConvolution struct {
	inputSize        [3]int            	// 3D size of the input/output data
	realKernSize     [3]int   			// Size of kernel and logical FFT size.
	fftKernLogicSize [3]int          	// logic size FFTed kernel, real parts only, we store less
	fftRBuf          [3]*data.Slice  	// FFT input buf; 2D:
	fftCBuf          [3]*data.Slice  	// FFT output buf; 2D:
	kern             [3]*data.Slice  	// FFT kernel on device
	cpukern          [3]*data.Slice  	// kernel on cpu
	fwPlan           fft3DR2CPlan   	// Forward FFT (1 component)
	bwPlan           fft3DC2RPlan    	// Backward FFT (1 component)

}

// Initializes a convolution to evaluate the scalar potential field for the given mesh geometry.
// TODO: add sanity check for this too
func NewPotential(inputSize, PBC [3]int, kernel [3]*data.Slice, test bool) *PotentialConvolution {
	c := new(PotentialConvolution)
	c.inputSize = inputSize
	c.realKernSize = kernel[X].Size()
	c.cpukern = kernel
	c.init(kernel)

	return c
}

// Calculate the scalar potential field of m * vol * Bsat, interpolate it, and add the result to the field.
// 	m:    magnetization normalized to unit length
// 	vol:  unitless mask used to scale m's length, may be nil
// 	Bsat: saturation magnetization in Tesla
func (c *PotentialConvolution) Exec3D(outp, inp, vol *data.Slice, mx, my, mz float64, targetGeom *data.Slice, includeOrExclude int, cellsize float64, Msat MSlice) {
	//zero1_async(c.ScalarPotential)
	util.Argument(inp.Size() == c.inputSize)
	for i := 0; i < 3; i++ { // FW FFT
		c.fwFFT(i, inp, vol, targetGeom, 1, includeOrExclude, Msat)
	}
	// kern mul
	Ext_move_spkernMulRSymm3D_async(c.fftCBuf, c.kern[X], c.kern[Y], c.kern[Z], c.fftKernLogicSize[X], c.fftKernLogicSize[Y], c.fftKernLogicSize[Z])

	//Larger buffer so we get enough points for preconditioning and interpolation
	size := c.inputSize
	size[X] += 8
	size[Y] += 8
	size[Z] += 8
	largebuffer := Buffer(1, size)
	defer Recycle(largebuffer)
	zero1_async(largebuffer)
	

	c.bwFFT(X, largebuffer, true, size)

	ext_move_cuinterp.Precondition(largebuffer, size)
	ext_move_cuinterp.InterpolateField(largebuffer, size, float32(mx), float32(my), float32(mz), 0, targetGeom, 2)
	Ext_move_Gradient_async(outp, largebuffer, c.inputSize, size, targetGeom, 1-includeOrExclude, float32(cellsize), 4)
}

//For eddy currents
func (c *PotentialConvolution) Exec3DSolenoidal(outp, inp, vol *data.Slice, periodicShift bool, useBsat int, Msat MSlice) {
	//zero1_async(c.ScalarPotential)
	for i := 0; i < 3; i++ { // FW FFT
		c.fwFFT(i, inp, vol, inp, useBsat, 2, Msat) //inp used as a dummy here at the third-to-last argument
	}
	// kern mul
	Ext_move_eddykernMulRSymm3D_async(c.fftCBuf, c.kern[X], c.kern[Y], c.kern[Z], c.fftKernLogicSize[X], c.fftKernLogicSize[Y], c.fftKernLogicSize[Z])

	for i := 0; i < 3; i++ { // BW FFT
		c.bwFFT(i, outp, periodicShift, c.inputSize)
	}
}

// forward FFT component i, with possibility to exclude or include only certain geometry and to use Bsat or not.
func (c *PotentialConvolution) fwFFT(i int, inp, vol *data.Slice, targetGeom *data.Slice, useBSat, includeOrExclude int, Msat MSlice) {
	zero1_async(c.fftRBuf[i])
	in := inp.Comp(i)
	Ext_move_copyPadMulGeom(c.fftRBuf[i], in, vol, c.realKernSize, c.inputSize, targetGeom, useBSat, includeOrExclude, Msat)
	c.fwPlan.ExecAsync(c.fftRBuf[i], c.fftCBuf[i])
}

// backward FFT component i, with possibility to exclude or include only certain geometry
func (c *PotentialConvolution) bwFFT(i int, outp *data.Slice, periodicShift bool, outpsize [3]int) {
	zero1_async(c.fftRBuf[i])
	c.bwPlan.ExecAsync(c.fftCBuf[i], c.fftRBuf[i])
	
	if(periodicShift) {
		out := outp.Comp(X)
		Ext_move_copyUnpadGeom(out, c.fftRBuf[i], outpsize, c.realKernSize, -4, 1, c.fftCBuf[i], 2) //c.fftCBuf[i] used here as a dummy for geometry, not really needed.
	} else {
		out := outp.Comp(i)
		Ext_move_copyUnpadGeom(out, c.fftRBuf[i], c.inputSize, c.realKernSize, 0, 0, c.fftCBuf[i], 2)
	}
}

func (c *PotentialConvolution) init(realKern [3]*data.Slice) {
	// init device buffers, this is always 3D by default

	nc := fftR2COutputSizeFloats(c.realKernSize)
	c.fftCBuf[X] = NewSlice(1, nc)
	c.fftCBuf[Y] = NewSlice(1, nc)
	c.fftCBuf[Z] = NewSlice(1, nc)

	c.fftRBuf[X] = NewSlice(1, c.realKernSize)
	c.fftRBuf[Y] = NewSlice(1, c.realKernSize)
	c.fftRBuf[Z] = NewSlice(1, c.realKernSize)

	// init FFT plans
	c.fwPlan = newFFT3DR2C(c.realKernSize[X], c.realKernSize[Y], c.realKernSize[Z])
	c.bwPlan = newFFT3DC2R(c.realKernSize[X], c.realKernSize[Y], c.realKernSize[Z])

	// init FFT kernel

	c.fftKernLogicSize = fftR2COutputSizeFloats(c.realKernSize)
	util.Assert(c.fftKernLogicSize[X]%2 == 0)
	c.fftKernLogicSize[X] /= 2

	// physical size of FFT(kernel): store only non-redundant part exploiting Y, Z mirror symmetry
	// X mirror symmetry already exploited: FFT(kernel) is purely imaginary.
	physKSize := [3]int{c.fftKernLogicSize[X], c.fftKernLogicSize[Y]/2 + 1, c.fftKernLogicSize[Z]/2 + 1}

	output := c.fftCBuf[0]
	input := c.fftRBuf[0]
	fftKern := data.NewSlice(1, physKSize)
	kfull := data.NewSlice(1, output.Size()) // not yet exploiting symmetry
	kfulls := kfull.Scalars()
	kCSize := physKSize
	kCSize[X] *= 2                     // size of kernel after removing Y,Z redundant parts, but still complex
	kCmplx := data.NewSlice(1, kCSize) // not yet exploiting X symmetry
	kc := kCmplx.Scalars()
	for i := 0; i < 3; i++ {
		if realKern[i] != nil { // ignore 0's
			// FW FFT
			data.Copy(input, realKern[i])
			c.fwPlan.ExecAsync(input, output)
			data.Copy(kfull, output)
			// extract non-redundant part (Y,Z symmetry)
			for iz := 0; iz < kCSize[Z]; iz++ {
				for iy := 0; iy < kCSize[Y]; iy++ {
					for ix := 0; ix < kCSize[X]; ix++ {
						kc[iz][iy][ix] = kfulls[iz][iy][ix]
					}
				}
			}
			scaleImaginaryParts(fftKern, kCmplx, 1.0/float32(c.fwPlan.InputLen())) //this kernel is imaginary, so take only those parts.
			c.kern[i] = GPUCopy(fftKern)
		}
	}
}

func (c *PotentialConvolution) Free() {
	if c == nil {
		return
	}
	c.inputSize = [3]int{}
	c.realKernSize = [3]int{}
	for i := 0; i < 3; i++ {
		c.fftCBuf[i].Free()
		c.fftRBuf[i].Free()
		c.fftCBuf[i] = nil
		c.fftRBuf[i] = nil

		c.kern[i].Free()
		c.kern[i] = nil
		
		c.fwPlan.Free()
		c.bwPlan.Free()
	}
}

// Extract imaginary parts, copy them from src to dst.
// In the meanwhile, check if real parts are nearly zero
// and scale the kernel to compensate for unnormalized FFTs.
// scale = 1/N, with N the FFT logical size.
func scaleImaginaryParts(dst, src *data.Slice, scale float32) {
	util.Log(dst.Len(), src.Len())
	util.Argument(2*dst.Len() == src.Len())
	util.Argument(dst.NComp() == 1 && src.NComp() == 1)

	srcList := src.Host()[0]
	dstList := dst.Host()[0]

	// Normally, the FFT'ed kernel is purely imaginary because of symmetry,
	// so we only store the imaginary parts...
	maxreal := float32(0.)
	for i := 0; i < src.Len()/2; i++ {
		dstList[i] = srcList[2*i+1] * scale
		if fabs(srcList[2*i]) > maxreal {
			maxreal = fabs(srcList[2*i])
		}
	}
	maxreal *= float32(math.Sqrt(float64(scale))) // after 1 FFT, normalization is sqrt(N)
	util.Log("FFT kernel real part: %v\n", maxreal)
	// ...however, we check that the real parts are nearly zero,
	// just to be sure we did not make a mistake during kernel creation.
	if maxreal > FFT_IMAG_TOLERANCE {
		log.Fatalf("FFT kernel real part: %v\n", maxreal)
	}
}
package ext_move_cuinterp

//#include "interpolationKernel.h"
import "C"

import (
	//"fmt"
	//"unsafe"
	"github.com/mumax/3/data"
	//"github.com/mumax/3/cuda/cu"
)

//Precondition float array into texture for spline interpolation
func Precondition(src *data.Slice, srcsize [3]int) {
	C.Precondition((*C.float)(src.DevPtr(0)), C.int(srcsize[0]), C.int(srcsize[1]), C.int(srcsize[2]))
}

//Skip preconditioning for e.g. linear interpolation
func NoPrecondition(src *data.Slice, srcsize [3]int) {
	C.NoPrecondition((*C.float)(src.DevPtr(0)), C.int(srcsize[0]), C.int(srcsize[1]), C.int(srcsize[2]))
}

//Interpolate field from texture into float array
func InterpolateField(dst *data.Slice, dstsize [3]int, mx, my, mz float32, cellshift int, geometry *data.Slice, includeOrExclude int) {
	C.InterpolateField((*C.float)(dst.DevPtr(0)), C.int(dstsize[0]), C.int(dstsize[1]), C.int(dstsize[2]), C.float(mx), C.float(my), C.float(mz), C.int(cellshift), (*C.float)(geometry.DevPtr(0)), C.int(includeOrExclude))	
}
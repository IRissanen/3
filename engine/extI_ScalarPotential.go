package engine

import (
	"bufio"
	"fmt"
	"github.com/mumax/3/data"
	"github.com/mumax/3/oommf"
	"github.com/mumax/3/timer"
	"github.com/mumax/3/util"
	"math"
	"os"
)

//The calculation of the scalar potential kernel, done in a similar fashion to the demagnetization tensor kernel calculation, except analytical integral is used.

const maxAspect = 100.0 // maximum sane cell aspect ratio, though as of yet scalar potential needs cubic cells anyway

func sanityCheck(cellsize [3]float64, pbc [3]int) {
	a3 := cellsize[X] / cellsize[Y]
	a2 := cellsize[Y] / cellsize[Z]
	a1 := cellsize[Z] / cellsize[X]

	if(!(a1 == a2 && a2 == a3)) {
		util.Fatal("Non-cubic cells, scalar potential analytic integration not valid:", cellsize)
	}
	aMax := math.Max(a1, math.Max(a2, a3))
	aMin := math.Min(a1, math.Min(a2, a3))

	if aMax > maxAspect || aMin < 1./maxAspect {
		util.Fatal("Unrealistic cell aspect ratio:", cellsize)
	}
}


// Obtains the demag kernel either from cacheDir/ or by calculating (and then storing in cacheDir for next time).
// Empty cacheDir disables caching.
func PotentialKernel(inputSize, pbc [3]int, cellsize [3]float64, accuracy float64, cacheDir string) (kernel [3]*data.Slice) {
	timer.Start("kernel_init")
	timer.Stop("kernel_init") // warm-up

	timer.Start("kernel_init")
	defer timer.Stop("kernel_init")

	sanityCheck(cellsize, pbc)
	// Cache disabled
	if cacheDir == "" {
		util.Log(`//Not using kernel cache (-cache="")`)
		return CalcScalarPotentialKernel(inputSize, pbc, cellsize, accuracy)
	}

	// Error-resilient kernel cache: if anything goes wrong, return calculated kernel.
	defer func() {
		if err := recover(); err != nil {
			util.Log("//Unable to use kernel cache:", err)
			kernel = CalcScalarPotentialKernel(inputSize, pbc, cellsize, accuracy)
		}
	}()

	// Try to load kernel
	basename := fmt.Sprint(cacheDir, "/", "mumax3kernel_", inputSize, "_", pbc, "_", cellsize, "_", accuracy, "_")
	var errLoad error
	for i := 0; i < 3; i++ {
			kernel[i], errLoad = LoadSPKernel(fmt.Sprint(basename, i, "S.ovf"))
			if errLoad != nil {
				break
			}
	}

	if errLoad != nil {
		util.Log("//Did not use cached kernel:", errLoad)
	} else {
		util.Log("//Using cached kernel:", basename)
		return kernel
	}

	// Could not load kernel: calculate it and save
	var errSave error
	kernel = CalcScalarPotentialKernel(inputSize, pbc, cellsize, accuracy)
	for i := 0; i < 3; i++ {
		
		errSave = SaveSPKernel(fmt.Sprint(basename, i, "S.ovf"), kernel[i])

		if errSave != nil {
			break
		}
	}
	if errSave != nil {
		util.Log("//Failed to cache kernel:", errSave)
	} else {
		util.Log("//Cached kernel:", basename)
	}

	return kernel
}

func LoadSPKernel(fname string) (kernel *data.Slice, err error) {
	kernel, _, err = oommf.ReadFile(fname)
	return
}

func SaveSPKernel(fname string, kernel *data.Slice) error {
	f, err := os.OpenFile(fname, os.O_WRONLY|os.O_TRUNC|os.O_CREATE, 0666)
	if err != nil {
		return err
	}
	out := bufio.NewWriter(f)
	defer out.Flush()
	oommf.WriteOVF2(out, kernel, data.Meta{}, "binary 4")
	return nil
}

//Analytical solution to the potential integral for cubic cells, taken from  C. Abert et al, IEEE Transactions on Magnetics 48, 1105 (2012)
func potentialIntegral(x,y,z float64) float64 {
	sqrtR := math.Sqrt(x*x+y*y+z*z)
	value := 0.0
	if(math.Abs(z) > 0 && sqrtR > 0){ //Do not want to divide by zero here
		value += -z*math.Atan(x*y/(z*sqrtR))
		value += y*math.Log(x+sqrtR)
		value += x*math.Log(y+sqrtR)
	}
	return value/(4.0*math.Pi)
}

func singleComponents(x,y,z float64, cellsize [3]float64) [3]float64 {
	sums := [3]float64{0.0,0.0,0.0}
	for i := -1; i <= 1; i+=2 {
		for j := -1; j <= 1; j+=2 {
			for k := -1; k <= 1; k+=2 {
				sums[Z] -= float64(i*j*k)*potentialIntegral(x+float64(i)*0.5*cellsize[X],y+float64(j)*0.5*cellsize[Y],z+float64(k)*0.5*cellsize[Z])
				sums[X] -= float64(i*j*k)*potentialIntegral(y+float64(i)*0.5*cellsize[Y],z+float64(j)*0.5*cellsize[Z],x+float64(k)*0.5*cellsize[X])
				sums[Y] -= float64(i*j*k)*potentialIntegral(z+float64(i)*0.5*cellsize[Z],x+float64(j)*0.5*cellsize[X],y+float64(k)*0.5*cellsize[Y])		
			}	
		}	
	}
	return sums
}

func CalcScalarPotentialKernel(inputSize, pbc [3]int, cellsize [3]float64, accuracy float64) (kernel [3]*data.Slice) {

	// Add zero-padding in non-PBC directions
	size := padSize(inputSize, pbc)
	var R [3]float64
	// Sanity check
	{
		util.Assert(size[Z] > 0 && size[Y] > 0 && size[X] > 0)
		util.Assert(cellsize[X] > 0 && cellsize[Y] > 0 && cellsize[Z] > 0)
		util.Assert(pbc[X] >= 0 && pbc[Y] >= 0 && pbc[Z] >= 0)
		util.Assert(accuracy > 0)
	}

	array64 := make([][][][]float64, 3)
	

	var array [3][][][]float32
	for c := 0; c < 3; c++ {
			kernel[c] = data.NewSlice(1, size)
			array[c] = kernel[c].Scalars()
		array64[c] = make([][][]float64, size[Z])
		for i := range array64[c] {
   	 		array64[c][i] = make([][]float64, size[Y])
    		for j := range array64[c][i] {
        		array64[c][i][j] = make([]float64, size[X])
    		}
		}
	}

	// Field (destination) loop ranges
	r1, r2 := kernelRanges(size, pbc)
	fmt.Println(r1, r2)
	for z := r1[Z]; z <= r2[Z]; z++ {
		zw := wrap(z, size[Z])
		if zw > size[Z]/2 {
			continue
		}
		R[Z] = (float64(z)) * cellsize[Z]

		for y := r1[Y]; y <= r2[Y]; y++ {

			yw := wrap(y, size[Y])
			if yw > size[Y]/2 {
				continue
			}
			R[Y] = (float64(y)) * cellsize[Y]
			for x := r1[X]; x <= r2[X]; x++ {
				xw := wrap(x, size[X])
				if xw > size[X]/2 {
					continue
				}
				R[X] = (float64(x)) * cellsize[X]
				comps := singleComponents(R[X], R[Y], R[Z], cellsize)
				array64[X][zw][yw][xw] += comps[X]
				array64[Y][zw][yw][xw] += comps[Y]
				array64[Z][zw][yw][xw] += comps[Z]
			}
		}
	}
	
	// Reconstruct skipped parts from symmetry (X)
	for z := 0; z < size[Z]; z++ {
		for y := 0; y < size[Y]; y++ {
			for x := size[X]/2+1; x < size[X]; x++ {
				x2 := size[X] - x
				array64[X][z][y][x] = -array64[X][z][y][x2]
				array64[Y][z][y][x] = array64[Y][z][y][x2]
				array64[Z][z][y][x] = array64[Z][z][y][x2]
			}
		}
	}

	
	// Reconstruct skipped parts from symmetry (Y)
	for z := 0; z < size[Z]; z++ {
		for y := size[Y]/2+1; y < size[Y]; y++ {
			y2 := size[Y] - y
			for x := 0; x < size[X]; x++ {
				array64[X][z][y][x] = array64[X][z][y2][x]
				array64[Y][z][y][x] = -array64[Y][z][y2][x]
				array64[Z][z][y][x] = array64[Z][z][y2][x]
			}
		}
	}	

	// Reconstruct skipped parts from symmetry (Z)
	for z := size[Z]/2+1; z < size[Z]; z++ {
		z2 := size[Z] - z
		for y := 0; y < size[Y]; y++ {
			for x := 0; x < size[X]; x++ {
				array64[X][z][y][x] = array64[X][z2][y][x]
				array64[Y][z][y][x] = array64[Y][z2][y][x]
				array64[Z][z][y][x] = -array64[Z][z2][y][x]
			}
		}
	}

	//Final conversion into float32s (in fact likely not needed, could've performed the calculation with float32s from the beginning)
	for z := 0; z < size[Z]; z++ {
		for y := 0; y < size[Y]; y++ {
			for x := 0; x < size[X]; x++ {
				array[X][z][y][x] = float32(array64[X][z][y][x])
				array[Y][z][y][x] = float32(array64[Y][z][y][x])
				array[Z][z][y][x] = float32(array64[Z][z][y][x])
			}
		}
	}	

	return kernel
}

//The three functions below are the same as for the demag kernel, just copied here because they cannot be used directly since demagkernel is in mag
func kernelRanges(size, pbc [3]int) (r1, r2 [3]int) {
	for c := 0; c < 3; c++ {
		if pbc[c] == 0 {
			r1[c], r2[c] = -(size[c]-1)/2, (size[c]-1)/2
		} else {
			r1[c], r2[c] = -(size[c]*pbc[c])+1, (size[c]*pbc[c])-1// no /2 here, or we would take half right and half left image
		}
	}
	return
}

func padSize(size, periodic [3]int) [3]int {
	var padded [3]int
	for i := range size {
		if periodic[i] != 0 {
			padded[i] = size[i]
			continue
		}
		if i != Z || size[i] > SMALL_N { 
			padded[i] = size[i] * 2
		} else {
			padded[i] = size[i]*2 - 1
		}
	}
	return padded
}

const SMALL_N = 5

func wrap(number, max int) int {
	for number < 0 {
		number += max
	}
	for number >= max {
		number -= max
	}
	return number
}
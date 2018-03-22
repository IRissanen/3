// Created by cgo - DO NOT EDIT

//line C:\Users\Ilari\Desktop\Jatko-opinnot\GoMumax\src\github.com\mumax\3\cuda\cuinterp\cuinterp.go:1
package cuinterp
//line C:\Users\Ilari\Desktop\Jatko-opinnot\GoMumax\src\github.com\mumax\3\cuda\cuinterp\cuinterp.go:7

//line C:\Users\Ilari\Desktop\Jatko-opinnot\GoMumax\src\github.com\mumax\3\cuda\cuinterp\cuinterp.go:6
import (
	"fmt"
	"unsafe"
	"github.com/mumax/3/data"
)
//line C:\Users\Ilari\Desktop\Jatko-opinnot\GoMumax\src\github.com\mumax\3\cuda\cuinterp\cuinterp.go:14

//line C:\Users\Ilari\Desktop\Jatko-opinnot\GoMumax\src\github.com\mumax\3\cuda\cuinterp\cuinterp.go:13
func InitCuda(newM, potential *data.Slice) {
	wub := unsafe.Pointer(newM)
	fmt.Println("derp", wub)
	Nx := 1
	Ny := 1
	Nz := 1
	mx := 1
	my := 1
	mz := 1
	_Cfunc_InterpolateAndGradient((*_Ctype_float)(unsafe.Pointer(newM[X])), (*_Ctype_float)(unsafe.Pointer(newM[Y])), (*_Ctype_float)(unsafe.Pointer(newM[Z])), (*_Ctype_float)(unsafe.Pointer(potential)), _Ctype_int(Nx), _Ctype_int(Ny), _Ctype_int(Nz), _Ctype_float(mx), _Ctype_float(my), _Ctype_float(mz))
}

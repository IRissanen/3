// Created by cgo - DO NOT EDIT

package cuinterp

import "unsafe"

import _ "runtime/cgo"

import "syscall"

var _ syscall.Errno
func _Cgo_ptr(ptr unsafe.Pointer) unsafe.Pointer { return ptr }

type _Ctype_float float32

type _Ctype_int int32

type _Ctype_void [0]byte

var _cgo_runtime_cgocall_errno func(unsafe.Pointer, uintptr) int32
var _cgo_runtime_cmalloc func(uintptr) unsafe.Pointer


var _cgo_738ea9aea9d3_Cfunc_InterpolateAndGradient unsafe.Pointer
func _Cfunc_InterpolateAndGradient(p0 *_Ctype_float, p1 *_Ctype_float, p2 *_Ctype_float, p3 *_Ctype_float, p4 _Ctype_int, p5 _Ctype_int, p6 _Ctype_int, p7 _Ctype_float, p8 _Ctype_float, p9 _Ctype_float) (r1 _Ctype_void) {
	_cgo_runtime_cgocall_errno(_cgo_738ea9aea9d3_Cfunc_InterpolateAndGradient, uintptr(unsafe.Pointer(&p0)))
	return
}

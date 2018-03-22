
#include "runtime.h"
#include "cgocall.h"
#include "textflag.h"

#pragma dataflag NOPTR
static void *cgocall_errno = runtime·cgocall_errno;
#pragma dataflag NOPTR
void *·_cgo_runtime_cgocall_errno = &cgocall_errno;

#pragma dataflag NOPTR
static void *runtime_gostring = runtime·gostring;
#pragma dataflag NOPTR
void *·_cgo_runtime_gostring = &runtime_gostring;

#pragma dataflag NOPTR
static void *runtime_gostringn = runtime·gostringn;
#pragma dataflag NOPTR
void *·_cgo_runtime_gostringn = &runtime_gostringn;

#pragma dataflag NOPTR
static void *runtime_gobytes = runtime·gobytes;
#pragma dataflag NOPTR
void *·_cgo_runtime_gobytes = &runtime_gobytes;

#pragma dataflag NOPTR
static void *runtime_cmalloc = runtime·cmalloc;
#pragma dataflag NOPTR
void *·_cgo_runtime_cmalloc = &runtime_cmalloc;

void ·_Cerrno(void*, int32);

#pragma cgo_import_static _cgo_738ea9aea9d3_Cfunc_InterpolateAndGradient
void _cgo_738ea9aea9d3_Cfunc_InterpolateAndGradient(void*);
#pragma dataflag NOPTR
void *·_cgo_738ea9aea9d3_Cfunc_InterpolateAndGradient = _cgo_738ea9aea9d3_Cfunc_InterpolateAndGradient;

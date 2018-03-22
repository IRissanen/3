#line 3 "C:\\Users\\Ilari\\Desktop\\Jatko-opinnot\\GoMumax\\src\\github.com\\mumax\\3\\cuda\\cuinterp\\cuinterp.go"
#include "capi.h"



// Usual nonsense: if x and y are not equal, the type will be invalid
// (have a negative array count) and an inscrutable error will come
// out of the compiler and hopefully mention "name".
#define __cgo_compile_assert_eq(x, y, name) typedef char name[(x-y)*(x-y)*-2+1];

// Check at compile time that the sizes we use match our expectations.
#define __cgo_size_assert(t, n) __cgo_compile_assert_eq(sizeof(t), n, _cgo_sizeof_##t##_is_not_##n)

__cgo_size_assert(char, 1)
__cgo_size_assert(short, 2)
__cgo_size_assert(int, 4)
typedef long long __cgo_long_long;
__cgo_size_assert(__cgo_long_long, 8)
__cgo_size_assert(float, 4)
__cgo_size_assert(double, 8)

extern char* _cgo_topofstack(void);

#include <errno.h>
#include <string.h>

void
_cgo_738ea9aea9d3_Cfunc_InterpolateAndGradient(void *v)
{
	struct {
		float* p0;
		float* p1;
		float* p2;
		float* p3;
		int p4;
		int p5;
		int p6;
		float p7;
		float p8;
		float p9;
	} __attribute__((__packed__, __gcc_struct__)) *a = v;
	InterpolateAndGradient((void*)a->p0, (void*)a->p1, (void*)a->p2, (void*)a->p3, a->p4, a->p5, a->p6, a->p7, a->p8, a->p9);
}


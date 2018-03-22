package ext_kernels

/*
 THIS FILE IS AUTO-GENERATED BY CUDA2GO.
 EDITING IS FUTILE.
*/

import(
	"unsafe"
	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/timer"
	"sync"
)

// CUDA handle for zeroOutsideGeometry kernel
var zeroOutsideGeometry_code cu.Function

// Stores the arguments for zeroOutsideGeometry kernel invocation
type zeroOutsideGeometry_args_t struct{
	 arg_EFieldX unsafe.Pointer
	 arg_EFieldY unsafe.Pointer
	 arg_EFieldZ unsafe.Pointer
	 arg_baseGeometry unsafe.Pointer
	 arg_sliderGeometry unsafe.Pointer
	 arg_Nx int
	 arg_Ny int
	 arg_Nz int
	 argptr [8]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for zeroOutsideGeometry kernel invocation
var zeroOutsideGeometry_args zeroOutsideGeometry_args_t

func init(){
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	 zeroOutsideGeometry_args.argptr[0] = unsafe.Pointer(&zeroOutsideGeometry_args.arg_EFieldX)
	 zeroOutsideGeometry_args.argptr[1] = unsafe.Pointer(&zeroOutsideGeometry_args.arg_EFieldY)
	 zeroOutsideGeometry_args.argptr[2] = unsafe.Pointer(&zeroOutsideGeometry_args.arg_EFieldZ)
	 zeroOutsideGeometry_args.argptr[3] = unsafe.Pointer(&zeroOutsideGeometry_args.arg_baseGeometry)
	 zeroOutsideGeometry_args.argptr[4] = unsafe.Pointer(&zeroOutsideGeometry_args.arg_sliderGeometry)
	 zeroOutsideGeometry_args.argptr[5] = unsafe.Pointer(&zeroOutsideGeometry_args.arg_Nx)
	 zeroOutsideGeometry_args.argptr[6] = unsafe.Pointer(&zeroOutsideGeometry_args.arg_Ny)
	 zeroOutsideGeometry_args.argptr[7] = unsafe.Pointer(&zeroOutsideGeometry_args.arg_Nz)
	 }

// Wrapper for zeroOutsideGeometry CUDA kernel, asynchronous.
func K_ext_zeroOutsideGeometry_async ( EFieldX unsafe.Pointer, EFieldY unsafe.Pointer, EFieldZ unsafe.Pointer, baseGeometry unsafe.Pointer, sliderGeometry unsafe.Pointer, Nx int, Ny int, Nz int,  cfg *config) {
	if Synchronous{ // debug
		Sync()
		timer.Start("zeroOutsideGeometry")
	}

	zeroOutsideGeometry_args.Lock()
	defer zeroOutsideGeometry_args.Unlock()

	if zeroOutsideGeometry_code == 0{
		zeroOutsideGeometry_code = fatbinLoad(zeroOutsideGeometry_map, "zeroOutsideGeometry")
	}

	 zeroOutsideGeometry_args.arg_EFieldX = EFieldX
	 zeroOutsideGeometry_args.arg_EFieldY = EFieldY
	 zeroOutsideGeometry_args.arg_EFieldZ = EFieldZ
	 zeroOutsideGeometry_args.arg_baseGeometry = baseGeometry
	 zeroOutsideGeometry_args.arg_sliderGeometry = sliderGeometry
	 zeroOutsideGeometry_args.arg_Nx = Nx
	 zeroOutsideGeometry_args.arg_Ny = Ny
	 zeroOutsideGeometry_args.arg_Nz = Nz
	

	args := zeroOutsideGeometry_args.argptr[:]
	cu.LaunchKernel(zeroOutsideGeometry_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous{ // debug
		Sync()
		timer.Stop("zeroOutsideGeometry")
	}
}

// maps compute capability on PTX code for zeroOutsideGeometry kernel.
var zeroOutsideGeometry_map = map[int]string{ 0: "" ,
20: zeroOutsideGeometry_ptx_20 ,
30: zeroOutsideGeometry_ptx_30 ,
35: zeroOutsideGeometry_ptx_35 ,
50: zeroOutsideGeometry_ptx_50 ,
52: zeroOutsideGeometry_ptx_52 ,
53: zeroOutsideGeometry_ptx_53  }

// zeroOutsideGeometry PTX code for various compute capabilities.
const(
  zeroOutsideGeometry_ptx_20 = `
.version 5.0
.target sm_20
.address_size 64

	// .globl	zeroOutsideGeometry

.visible .entry zeroOutsideGeometry(
	.param .u64 zeroOutsideGeometry_param_0,
	.param .u64 zeroOutsideGeometry_param_1,
	.param .u64 zeroOutsideGeometry_param_2,
	.param .u64 zeroOutsideGeometry_param_3,
	.param .u64 zeroOutsideGeometry_param_4,
	.param .u32 zeroOutsideGeometry_param_5,
	.param .u32 zeroOutsideGeometry_param_6,
	.param .u32 zeroOutsideGeometry_param_7
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<19>;
	.reg .b64 	%rd<20>;


	ld.param.u64 	%rd2, [zeroOutsideGeometry_param_0];
	ld.param.u64 	%rd3, [zeroOutsideGeometry_param_1];
	ld.param.u64 	%rd4, [zeroOutsideGeometry_param_2];
	ld.param.u64 	%rd5, [zeroOutsideGeometry_param_3];
	ld.param.u64 	%rd6, [zeroOutsideGeometry_param_4];
	ld.param.u32 	%r4, [zeroOutsideGeometry_param_5];
	ld.param.u32 	%r5, [zeroOutsideGeometry_param_6];
	ld.param.u32 	%r6, [zeroOutsideGeometry_param_7];
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %ctaid.x;
	mov.u32 	%r9, %tid.x;
	mad.lo.s32 	%r1, %r7, %r8, %r9;
	mov.u32 	%r10, %ntid.y;
	mov.u32 	%r11, %ctaid.y;
	mov.u32 	%r12, %tid.y;
	mad.lo.s32 	%r2, %r10, %r11, %r12;
	mov.u32 	%r13, %ntid.z;
	mov.u32 	%r14, %ctaid.z;
	mov.u32 	%r15, %tid.z;
	mad.lo.s32 	%r3, %r13, %r14, %r15;
	setp.ge.s32	%p1, %r2, %r5;
	setp.ge.s32	%p2, %r1, %r4;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32	%p4, %r3, %r6;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	BB0_4;

	cvta.to.global.u64 	%rd7, %rd6;
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	cvt.s64.s32	%rd1, %r17;
	mul.wide.s32 	%rd8, %r17, 4;
	add.s64 	%rd9, %rd7, %rd8;
	ld.global.f32 	%f1, [%rd9];
	setp.eq.f32	%p6, %f1, 0f40000000;
	@%p6 bra 	BB0_4;

	cvta.to.global.u64 	%rd10, %rd5;
	shl.b64 	%rd11, %rd1, 2;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.f32 	%f2, [%rd12];
	setp.eq.f32	%p7, %f2, 0f40000000;
	@%p7 bra 	BB0_4;

	cvta.to.global.u64 	%rd13, %rd4;
	cvta.to.global.u64 	%rd14, %rd3;
	cvta.to.global.u64 	%rd15, %rd2;
	add.s64 	%rd17, %rd15, %rd11;
	mov.u32 	%r18, 0;
	st.global.u32 	[%rd17], %r18;
	add.s64 	%rd18, %rd14, %rd11;
	st.global.u32 	[%rd18], %r18;
	add.s64 	%rd19, %rd13, %rd11;
	st.global.u32 	[%rd19], %r18;

BB0_4:
	ret;
}


`
   zeroOutsideGeometry_ptx_30 = `
.version 5.0
.target sm_30
.address_size 64

	// .globl	zeroOutsideGeometry

.visible .entry zeroOutsideGeometry(
	.param .u64 zeroOutsideGeometry_param_0,
	.param .u64 zeroOutsideGeometry_param_1,
	.param .u64 zeroOutsideGeometry_param_2,
	.param .u64 zeroOutsideGeometry_param_3,
	.param .u64 zeroOutsideGeometry_param_4,
	.param .u32 zeroOutsideGeometry_param_5,
	.param .u32 zeroOutsideGeometry_param_6,
	.param .u32 zeroOutsideGeometry_param_7
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<19>;
	.reg .b64 	%rd<20>;


	ld.param.u64 	%rd2, [zeroOutsideGeometry_param_0];
	ld.param.u64 	%rd3, [zeroOutsideGeometry_param_1];
	ld.param.u64 	%rd4, [zeroOutsideGeometry_param_2];
	ld.param.u64 	%rd5, [zeroOutsideGeometry_param_3];
	ld.param.u64 	%rd6, [zeroOutsideGeometry_param_4];
	ld.param.u32 	%r4, [zeroOutsideGeometry_param_5];
	ld.param.u32 	%r5, [zeroOutsideGeometry_param_6];
	ld.param.u32 	%r6, [zeroOutsideGeometry_param_7];
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %ctaid.x;
	mov.u32 	%r9, %tid.x;
	mad.lo.s32 	%r1, %r7, %r8, %r9;
	mov.u32 	%r10, %ntid.y;
	mov.u32 	%r11, %ctaid.y;
	mov.u32 	%r12, %tid.y;
	mad.lo.s32 	%r2, %r10, %r11, %r12;
	mov.u32 	%r13, %ntid.z;
	mov.u32 	%r14, %ctaid.z;
	mov.u32 	%r15, %tid.z;
	mad.lo.s32 	%r3, %r13, %r14, %r15;
	setp.ge.s32	%p1, %r2, %r5;
	setp.ge.s32	%p2, %r1, %r4;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32	%p4, %r3, %r6;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	BB0_4;

	cvta.to.global.u64 	%rd7, %rd6;
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	cvt.s64.s32	%rd1, %r17;
	mul.wide.s32 	%rd8, %r17, 4;
	add.s64 	%rd9, %rd7, %rd8;
	ld.global.f32 	%f1, [%rd9];
	setp.eq.f32	%p6, %f1, 0f40000000;
	@%p6 bra 	BB0_4;

	cvta.to.global.u64 	%rd10, %rd5;
	shl.b64 	%rd11, %rd1, 2;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.f32 	%f2, [%rd12];
	setp.eq.f32	%p7, %f2, 0f40000000;
	@%p7 bra 	BB0_4;

	cvta.to.global.u64 	%rd13, %rd4;
	cvta.to.global.u64 	%rd14, %rd3;
	cvta.to.global.u64 	%rd15, %rd2;
	add.s64 	%rd17, %rd15, %rd11;
	mov.u32 	%r18, 0;
	st.global.u32 	[%rd17], %r18;
	add.s64 	%rd18, %rd14, %rd11;
	st.global.u32 	[%rd18], %r18;
	add.s64 	%rd19, %rd13, %rd11;
	st.global.u32 	[%rd19], %r18;

BB0_4:
	ret;
}


`
   zeroOutsideGeometry_ptx_35 = `
.version 5.0
.target sm_35
.address_size 64

	// .globl	zeroOutsideGeometry

.visible .entry zeroOutsideGeometry(
	.param .u64 zeroOutsideGeometry_param_0,
	.param .u64 zeroOutsideGeometry_param_1,
	.param .u64 zeroOutsideGeometry_param_2,
	.param .u64 zeroOutsideGeometry_param_3,
	.param .u64 zeroOutsideGeometry_param_4,
	.param .u32 zeroOutsideGeometry_param_5,
	.param .u32 zeroOutsideGeometry_param_6,
	.param .u32 zeroOutsideGeometry_param_7
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<19>;
	.reg .b64 	%rd<20>;


	ld.param.u64 	%rd2, [zeroOutsideGeometry_param_0];
	ld.param.u64 	%rd3, [zeroOutsideGeometry_param_1];
	ld.param.u64 	%rd4, [zeroOutsideGeometry_param_2];
	ld.param.u64 	%rd5, [zeroOutsideGeometry_param_3];
	ld.param.u64 	%rd6, [zeroOutsideGeometry_param_4];
	ld.param.u32 	%r4, [zeroOutsideGeometry_param_5];
	ld.param.u32 	%r5, [zeroOutsideGeometry_param_6];
	ld.param.u32 	%r6, [zeroOutsideGeometry_param_7];
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %ctaid.x;
	mov.u32 	%r9, %tid.x;
	mad.lo.s32 	%r1, %r7, %r8, %r9;
	mov.u32 	%r10, %ntid.y;
	mov.u32 	%r11, %ctaid.y;
	mov.u32 	%r12, %tid.y;
	mad.lo.s32 	%r2, %r10, %r11, %r12;
	mov.u32 	%r13, %ntid.z;
	mov.u32 	%r14, %ctaid.z;
	mov.u32 	%r15, %tid.z;
	mad.lo.s32 	%r3, %r13, %r14, %r15;
	setp.ge.s32	%p1, %r2, %r5;
	setp.ge.s32	%p2, %r1, %r4;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32	%p4, %r3, %r6;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	BB0_4;

	cvta.to.global.u64 	%rd7, %rd6;
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	cvt.s64.s32	%rd1, %r17;
	mul.wide.s32 	%rd8, %r17, 4;
	add.s64 	%rd9, %rd7, %rd8;
	ld.global.nc.f32 	%f1, [%rd9];
	setp.eq.f32	%p6, %f1, 0f40000000;
	@%p6 bra 	BB0_4;

	cvta.to.global.u64 	%rd10, %rd5;
	shl.b64 	%rd11, %rd1, 2;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.nc.f32 	%f2, [%rd12];
	setp.eq.f32	%p7, %f2, 0f40000000;
	@%p7 bra 	BB0_4;

	cvta.to.global.u64 	%rd13, %rd4;
	cvta.to.global.u64 	%rd14, %rd3;
	cvta.to.global.u64 	%rd15, %rd2;
	add.s64 	%rd17, %rd15, %rd11;
	mov.u32 	%r18, 0;
	st.global.u32 	[%rd17], %r18;
	add.s64 	%rd18, %rd14, %rd11;
	st.global.u32 	[%rd18], %r18;
	add.s64 	%rd19, %rd13, %rd11;
	st.global.u32 	[%rd19], %r18;

BB0_4:
	ret;
}


`
   zeroOutsideGeometry_ptx_50 = `
.version 5.0
.target sm_50
.address_size 64

	// .globl	zeroOutsideGeometry

.visible .entry zeroOutsideGeometry(
	.param .u64 zeroOutsideGeometry_param_0,
	.param .u64 zeroOutsideGeometry_param_1,
	.param .u64 zeroOutsideGeometry_param_2,
	.param .u64 zeroOutsideGeometry_param_3,
	.param .u64 zeroOutsideGeometry_param_4,
	.param .u32 zeroOutsideGeometry_param_5,
	.param .u32 zeroOutsideGeometry_param_6,
	.param .u32 zeroOutsideGeometry_param_7
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<19>;
	.reg .b64 	%rd<20>;


	ld.param.u64 	%rd2, [zeroOutsideGeometry_param_0];
	ld.param.u64 	%rd3, [zeroOutsideGeometry_param_1];
	ld.param.u64 	%rd4, [zeroOutsideGeometry_param_2];
	ld.param.u64 	%rd5, [zeroOutsideGeometry_param_3];
	ld.param.u64 	%rd6, [zeroOutsideGeometry_param_4];
	ld.param.u32 	%r4, [zeroOutsideGeometry_param_5];
	ld.param.u32 	%r5, [zeroOutsideGeometry_param_6];
	ld.param.u32 	%r6, [zeroOutsideGeometry_param_7];
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %ctaid.x;
	mov.u32 	%r9, %tid.x;
	mad.lo.s32 	%r1, %r7, %r8, %r9;
	mov.u32 	%r10, %ntid.y;
	mov.u32 	%r11, %ctaid.y;
	mov.u32 	%r12, %tid.y;
	mad.lo.s32 	%r2, %r10, %r11, %r12;
	mov.u32 	%r13, %ntid.z;
	mov.u32 	%r14, %ctaid.z;
	mov.u32 	%r15, %tid.z;
	mad.lo.s32 	%r3, %r13, %r14, %r15;
	setp.ge.s32	%p1, %r2, %r5;
	setp.ge.s32	%p2, %r1, %r4;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32	%p4, %r3, %r6;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	BB0_4;

	cvta.to.global.u64 	%rd7, %rd6;
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	cvt.s64.s32	%rd1, %r17;
	mul.wide.s32 	%rd8, %r17, 4;
	add.s64 	%rd9, %rd7, %rd8;
	ld.global.nc.f32 	%f1, [%rd9];
	setp.eq.f32	%p6, %f1, 0f40000000;
	@%p6 bra 	BB0_4;

	cvta.to.global.u64 	%rd10, %rd5;
	shl.b64 	%rd11, %rd1, 2;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.nc.f32 	%f2, [%rd12];
	setp.eq.f32	%p7, %f2, 0f40000000;
	@%p7 bra 	BB0_4;

	cvta.to.global.u64 	%rd13, %rd4;
	cvta.to.global.u64 	%rd14, %rd3;
	cvta.to.global.u64 	%rd15, %rd2;
	add.s64 	%rd17, %rd15, %rd11;
	mov.u32 	%r18, 0;
	st.global.u32 	[%rd17], %r18;
	add.s64 	%rd18, %rd14, %rd11;
	st.global.u32 	[%rd18], %r18;
	add.s64 	%rd19, %rd13, %rd11;
	st.global.u32 	[%rd19], %r18;

BB0_4:
	ret;
}


`
   zeroOutsideGeometry_ptx_52 = `
.version 5.0
.target sm_52
.address_size 64

	// .globl	zeroOutsideGeometry

.visible .entry zeroOutsideGeometry(
	.param .u64 zeroOutsideGeometry_param_0,
	.param .u64 zeroOutsideGeometry_param_1,
	.param .u64 zeroOutsideGeometry_param_2,
	.param .u64 zeroOutsideGeometry_param_3,
	.param .u64 zeroOutsideGeometry_param_4,
	.param .u32 zeroOutsideGeometry_param_5,
	.param .u32 zeroOutsideGeometry_param_6,
	.param .u32 zeroOutsideGeometry_param_7
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<19>;
	.reg .b64 	%rd<20>;


	ld.param.u64 	%rd2, [zeroOutsideGeometry_param_0];
	ld.param.u64 	%rd3, [zeroOutsideGeometry_param_1];
	ld.param.u64 	%rd4, [zeroOutsideGeometry_param_2];
	ld.param.u64 	%rd5, [zeroOutsideGeometry_param_3];
	ld.param.u64 	%rd6, [zeroOutsideGeometry_param_4];
	ld.param.u32 	%r4, [zeroOutsideGeometry_param_5];
	ld.param.u32 	%r5, [zeroOutsideGeometry_param_6];
	ld.param.u32 	%r6, [zeroOutsideGeometry_param_7];
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %ctaid.x;
	mov.u32 	%r9, %tid.x;
	mad.lo.s32 	%r1, %r7, %r8, %r9;
	mov.u32 	%r10, %ntid.y;
	mov.u32 	%r11, %ctaid.y;
	mov.u32 	%r12, %tid.y;
	mad.lo.s32 	%r2, %r10, %r11, %r12;
	mov.u32 	%r13, %ntid.z;
	mov.u32 	%r14, %ctaid.z;
	mov.u32 	%r15, %tid.z;
	mad.lo.s32 	%r3, %r13, %r14, %r15;
	setp.ge.s32	%p1, %r2, %r5;
	setp.ge.s32	%p2, %r1, %r4;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32	%p4, %r3, %r6;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	BB0_4;

	cvta.to.global.u64 	%rd7, %rd6;
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	cvt.s64.s32	%rd1, %r17;
	mul.wide.s32 	%rd8, %r17, 4;
	add.s64 	%rd9, %rd7, %rd8;
	ld.global.nc.f32 	%f1, [%rd9];
	setp.eq.f32	%p6, %f1, 0f40000000;
	@%p6 bra 	BB0_4;

	cvta.to.global.u64 	%rd10, %rd5;
	shl.b64 	%rd11, %rd1, 2;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.nc.f32 	%f2, [%rd12];
	setp.eq.f32	%p7, %f2, 0f40000000;
	@%p7 bra 	BB0_4;

	cvta.to.global.u64 	%rd13, %rd4;
	cvta.to.global.u64 	%rd14, %rd3;
	cvta.to.global.u64 	%rd15, %rd2;
	add.s64 	%rd17, %rd15, %rd11;
	mov.u32 	%r18, 0;
	st.global.u32 	[%rd17], %r18;
	add.s64 	%rd18, %rd14, %rd11;
	st.global.u32 	[%rd18], %r18;
	add.s64 	%rd19, %rd13, %rd11;
	st.global.u32 	[%rd19], %r18;

BB0_4:
	ret;
}


`
   zeroOutsideGeometry_ptx_53 = `
.version 5.0
.target sm_20
.address_size 64

	// .globl	zeroOutsideGeometry

.visible .entry zeroOutsideGeometry(
	.param .u64 zeroOutsideGeometry_param_0,
	.param .u64 zeroOutsideGeometry_param_1,
	.param .u64 zeroOutsideGeometry_param_2,
	.param .u64 zeroOutsideGeometry_param_3,
	.param .u64 zeroOutsideGeometry_param_4,
	.param .u32 zeroOutsideGeometry_param_5,
	.param .u32 zeroOutsideGeometry_param_6,
	.param .u32 zeroOutsideGeometry_param_7
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<19>;
	.reg .b64 	%rd<20>;


	ld.param.u64 	%rd2, [zeroOutsideGeometry_param_0];
	ld.param.u64 	%rd3, [zeroOutsideGeometry_param_1];
	ld.param.u64 	%rd4, [zeroOutsideGeometry_param_2];
	ld.param.u64 	%rd5, [zeroOutsideGeometry_param_3];
	ld.param.u64 	%rd6, [zeroOutsideGeometry_param_4];
	ld.param.u32 	%r4, [zeroOutsideGeometry_param_5];
	ld.param.u32 	%r5, [zeroOutsideGeometry_param_6];
	ld.param.u32 	%r6, [zeroOutsideGeometry_param_7];
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %ctaid.x;
	mov.u32 	%r9, %tid.x;
	mad.lo.s32 	%r1, %r7, %r8, %r9;
	mov.u32 	%r10, %ntid.y;
	mov.u32 	%r11, %ctaid.y;
	mov.u32 	%r12, %tid.y;
	mad.lo.s32 	%r2, %r10, %r11, %r12;
	mov.u32 	%r13, %ntid.z;
	mov.u32 	%r14, %ctaid.z;
	mov.u32 	%r15, %tid.z;
	mad.lo.s32 	%r3, %r13, %r14, %r15;
	setp.ge.s32	%p1, %r2, %r5;
	setp.ge.s32	%p2, %r1, %r4;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32	%p4, %r3, %r6;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	BB0_4;

	cvta.to.global.u64 	%rd7, %rd6;
	mad.lo.s32 	%r16, %r3, %r5, %r2;
	mad.lo.s32 	%r17, %r16, %r4, %r1;
	cvt.s64.s32	%rd1, %r17;
	mul.wide.s32 	%rd8, %r17, 4;
	add.s64 	%rd9, %rd7, %rd8;
	ld.global.f32 	%f1, [%rd9];
	setp.eq.f32	%p6, %f1, 0f40000000;
	@%p6 bra 	BB0_4;

	cvta.to.global.u64 	%rd10, %rd5;
	shl.b64 	%rd11, %rd1, 2;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.f32 	%f2, [%rd12];
	setp.eq.f32	%p7, %f2, 0f40000000;
	@%p7 bra 	BB0_4;

	cvta.to.global.u64 	%rd13, %rd4;
	cvta.to.global.u64 	%rd14, %rd3;
	cvta.to.global.u64 	%rd15, %rd2;
	add.s64 	%rd17, %rd15, %rd11;
	mov.u32 	%r18, 0;
	st.global.u32 	[%rd17], %r18;
	add.s64 	%rd18, %rd14, %rd11;
	st.global.u32 	[%rd18], %r18;
	add.s64 	%rd19, %rd13, %rd11;
	st.global.u32 	[%rd19], %r18;

BB0_4:
	ret;
}


`
 )

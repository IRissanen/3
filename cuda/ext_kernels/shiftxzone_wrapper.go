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

// CUDA handle for shiftxzone kernel
var shiftxzone_code cu.Function

// Stores the arguments for shiftxzone kernel invocation
type shiftxzone_args_t struct{
	 arg_dst unsafe.Pointer
	 arg_src unsafe.Pointer
	 arg_Nx int
	 arg_Ny int
	 arg_Nz int
	 arg_shx int
	 arg_shy int
	 arg_shz int
	 arg_clampL float32
	 arg_clampR float32
	 arg_zoneGeom unsafe.Pointer
	 argptr [11]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for shiftxzone kernel invocation
var shiftxzone_args shiftxzone_args_t

func init(){
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	 shiftxzone_args.argptr[0] = unsafe.Pointer(&shiftxzone_args.arg_dst)
	 shiftxzone_args.argptr[1] = unsafe.Pointer(&shiftxzone_args.arg_src)
	 shiftxzone_args.argptr[2] = unsafe.Pointer(&shiftxzone_args.arg_Nx)
	 shiftxzone_args.argptr[3] = unsafe.Pointer(&shiftxzone_args.arg_Ny)
	 shiftxzone_args.argptr[4] = unsafe.Pointer(&shiftxzone_args.arg_Nz)
	 shiftxzone_args.argptr[5] = unsafe.Pointer(&shiftxzone_args.arg_shx)
	 shiftxzone_args.argptr[6] = unsafe.Pointer(&shiftxzone_args.arg_shy)
	 shiftxzone_args.argptr[7] = unsafe.Pointer(&shiftxzone_args.arg_shz)
	 shiftxzone_args.argptr[8] = unsafe.Pointer(&shiftxzone_args.arg_clampL)
	 shiftxzone_args.argptr[9] = unsafe.Pointer(&shiftxzone_args.arg_clampR)
	 shiftxzone_args.argptr[10] = unsafe.Pointer(&shiftxzone_args.arg_zoneGeom)
	 }

// Wrapper for shiftxzone CUDA kernel, asynchronous.
func K_ext_shiftxzone_async ( dst unsafe.Pointer, src unsafe.Pointer, Nx int, Ny int, Nz int, shx int, shy int, shz int, clampL float32, clampR float32, zoneGeom unsafe.Pointer,  cfg *config) {
	if Synchronous{ // debug
		Sync()
		timer.Start("shiftxzone")
	}

	shiftxzone_args.Lock()
	defer shiftxzone_args.Unlock()

	if shiftxzone_code == 0{
		shiftxzone_code = fatbinLoad(shiftxzone_map, "shiftxzone")
	}

	 shiftxzone_args.arg_dst = dst
	 shiftxzone_args.arg_src = src
	 shiftxzone_args.arg_Nx = Nx
	 shiftxzone_args.arg_Ny = Ny
	 shiftxzone_args.arg_Nz = Nz
	 shiftxzone_args.arg_shx = shx
	 shiftxzone_args.arg_shy = shy
	 shiftxzone_args.arg_shz = shz
	 shiftxzone_args.arg_clampL = clampL
	 shiftxzone_args.arg_clampR = clampR
	 shiftxzone_args.arg_zoneGeom = zoneGeom
	

	args := shiftxzone_args.argptr[:]
	cu.LaunchKernel(shiftxzone_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous{ // debug
		Sync()
		timer.Stop("shiftxzone")
	}
}

// maps compute capability on PTX code for shiftxzone kernel.
var shiftxzone_map = map[int]string{ 0: "" ,
20: shiftxzone_ptx_20 ,
30: shiftxzone_ptx_30 ,
35: shiftxzone_ptx_35 ,
50: shiftxzone_ptx_50 ,
52: shiftxzone_ptx_52 ,
53: shiftxzone_ptx_53  }

// shiftxzone PTX code for various compute capabilities.
const(
  shiftxzone_ptx_20 = `
.version 5.0
.target sm_20
.address_size 64

	// .globl	shiftxzone

.visible .entry shiftxzone(
	.param .u64 shiftxzone_param_0,
	.param .u64 shiftxzone_param_1,
	.param .u32 shiftxzone_param_2,
	.param .u32 shiftxzone_param_3,
	.param .u32 shiftxzone_param_4,
	.param .u32 shiftxzone_param_5,
	.param .u32 shiftxzone_param_6,
	.param .u32 shiftxzone_param_7,
	.param .f32 shiftxzone_param_8,
	.param .f32 shiftxzone_param_9,
	.param .u64 shiftxzone_param_10
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<32>;
	.reg .b64 	%rd<16>;


	ld.param.u64 	%rd3, [shiftxzone_param_0];
	ld.param.u64 	%rd4, [shiftxzone_param_1];
	ld.param.u32 	%r5, [shiftxzone_param_2];
	ld.param.u32 	%r6, [shiftxzone_param_3];
	ld.param.u32 	%r7, [shiftxzone_param_4];
	ld.param.u32 	%r8, [shiftxzone_param_5];
	ld.param.u32 	%r9, [shiftxzone_param_6];
	ld.param.u32 	%r10, [shiftxzone_param_7];
	ld.param.u64 	%rd5, [shiftxzone_param_10];
	cvta.to.global.u64 	%rd1, %rd5;
	mov.u32 	%r11, %ntid.x;
	mov.u32 	%r12, %ctaid.x;
	mov.u32 	%r13, %tid.x;
	mad.lo.s32 	%r1, %r11, %r12, %r13;
	mov.u32 	%r14, %ntid.y;
	mov.u32 	%r15, %ctaid.y;
	mov.u32 	%r16, %tid.y;
	mad.lo.s32 	%r2, %r14, %r15, %r16;
	mov.u32 	%r17, %ntid.z;
	mov.u32 	%r18, %ctaid.z;
	mov.u32 	%r19, %tid.z;
	mad.lo.s32 	%r3, %r17, %r18, %r19;
	setp.ge.s32	%p1, %r2, %r6;
	setp.ge.s32	%p2, %r1, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	BB0_4;

	mad.lo.s32 	%r20, %r3, %r6, %r2;
	mad.lo.s32 	%r21, %r20, %r5, %r1;
	add.s32 	%r22, %r8, %r5;
	add.s32 	%r23, %r22, %r1;
	rem.s32 	%r24, %r23, %r5;
	add.s32 	%r25, %r9, %r6;
	add.s32 	%r26, %r25, %r2;
	rem.s32 	%r27, %r26, %r6;
	add.s32 	%r28, %r10, %r7;
	add.s32 	%r29, %r28, %r3;
	rem.s32 	%r30, %r29, %r7;
	mad.lo.s32 	%r31, %r30, %r6, %r27;
	mad.lo.s32 	%r4, %r31, %r5, %r24;
	cvt.s64.s32	%rd2, %r21;
	mul.wide.s32 	%rd6, %r21, 4;
	add.s64 	%rd7, %rd1, %rd6;
	ld.global.f32 	%f1, [%rd7];
	setp.neu.f32	%p6, %f1, 0f00000000;
	@%p6 bra 	BB0_3;

	mul.wide.s32 	%rd8, %r4, 4;
	add.s64 	%rd9, %rd1, %rd8;
	ld.global.f32 	%f2, [%rd9];
	setp.eq.f32	%p7, %f2, 0f00000000;
	@%p7 bra 	BB0_4;

BB0_3:
	cvta.to.global.u64 	%rd10, %rd3;
	cvta.to.global.u64 	%rd11, %rd4;
	shl.b64 	%rd12, %rd2, 2;
	add.s64 	%rd13, %rd11, %rd12;
	ld.global.f32 	%f3, [%rd13];
	mul.wide.s32 	%rd14, %r4, 4;
	add.s64 	%rd15, %rd10, %rd14;
	st.global.f32 	[%rd15], %f3;

BB0_4:
	ret;
}


`
   shiftxzone_ptx_30 = `
.version 5.0
.target sm_30
.address_size 64

	// .globl	shiftxzone

.visible .entry shiftxzone(
	.param .u64 shiftxzone_param_0,
	.param .u64 shiftxzone_param_1,
	.param .u32 shiftxzone_param_2,
	.param .u32 shiftxzone_param_3,
	.param .u32 shiftxzone_param_4,
	.param .u32 shiftxzone_param_5,
	.param .u32 shiftxzone_param_6,
	.param .u32 shiftxzone_param_7,
	.param .f32 shiftxzone_param_8,
	.param .f32 shiftxzone_param_9,
	.param .u64 shiftxzone_param_10
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<32>;
	.reg .b64 	%rd<16>;


	ld.param.u64 	%rd3, [shiftxzone_param_0];
	ld.param.u64 	%rd4, [shiftxzone_param_1];
	ld.param.u32 	%r5, [shiftxzone_param_2];
	ld.param.u32 	%r6, [shiftxzone_param_3];
	ld.param.u32 	%r7, [shiftxzone_param_4];
	ld.param.u32 	%r8, [shiftxzone_param_5];
	ld.param.u32 	%r9, [shiftxzone_param_6];
	ld.param.u32 	%r10, [shiftxzone_param_7];
	ld.param.u64 	%rd5, [shiftxzone_param_10];
	cvta.to.global.u64 	%rd1, %rd5;
	mov.u32 	%r11, %ntid.x;
	mov.u32 	%r12, %ctaid.x;
	mov.u32 	%r13, %tid.x;
	mad.lo.s32 	%r1, %r11, %r12, %r13;
	mov.u32 	%r14, %ntid.y;
	mov.u32 	%r15, %ctaid.y;
	mov.u32 	%r16, %tid.y;
	mad.lo.s32 	%r2, %r14, %r15, %r16;
	mov.u32 	%r17, %ntid.z;
	mov.u32 	%r18, %ctaid.z;
	mov.u32 	%r19, %tid.z;
	mad.lo.s32 	%r3, %r17, %r18, %r19;
	setp.ge.s32	%p1, %r2, %r6;
	setp.ge.s32	%p2, %r1, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	BB0_4;

	mad.lo.s32 	%r20, %r3, %r6, %r2;
	mad.lo.s32 	%r21, %r20, %r5, %r1;
	add.s32 	%r22, %r8, %r5;
	add.s32 	%r23, %r22, %r1;
	rem.s32 	%r24, %r23, %r5;
	add.s32 	%r25, %r9, %r6;
	add.s32 	%r26, %r25, %r2;
	rem.s32 	%r27, %r26, %r6;
	add.s32 	%r28, %r10, %r7;
	add.s32 	%r29, %r28, %r3;
	rem.s32 	%r30, %r29, %r7;
	mad.lo.s32 	%r31, %r30, %r6, %r27;
	mad.lo.s32 	%r4, %r31, %r5, %r24;
	cvt.s64.s32	%rd2, %r21;
	mul.wide.s32 	%rd6, %r21, 4;
	add.s64 	%rd7, %rd1, %rd6;
	ld.global.f32 	%f1, [%rd7];
	setp.neu.f32	%p6, %f1, 0f00000000;
	@%p6 bra 	BB0_3;

	mul.wide.s32 	%rd8, %r4, 4;
	add.s64 	%rd9, %rd1, %rd8;
	ld.global.f32 	%f2, [%rd9];
	setp.eq.f32	%p7, %f2, 0f00000000;
	@%p7 bra 	BB0_4;

BB0_3:
	cvta.to.global.u64 	%rd10, %rd3;
	cvta.to.global.u64 	%rd11, %rd4;
	shl.b64 	%rd12, %rd2, 2;
	add.s64 	%rd13, %rd11, %rd12;
	ld.global.f32 	%f3, [%rd13];
	mul.wide.s32 	%rd14, %r4, 4;
	add.s64 	%rd15, %rd10, %rd14;
	st.global.f32 	[%rd15], %f3;

BB0_4:
	ret;
}


`
   shiftxzone_ptx_35 = `
.version 5.0
.target sm_35
.address_size 64

	// .globl	shiftxzone

.visible .entry shiftxzone(
	.param .u64 shiftxzone_param_0,
	.param .u64 shiftxzone_param_1,
	.param .u32 shiftxzone_param_2,
	.param .u32 shiftxzone_param_3,
	.param .u32 shiftxzone_param_4,
	.param .u32 shiftxzone_param_5,
	.param .u32 shiftxzone_param_6,
	.param .u32 shiftxzone_param_7,
	.param .f32 shiftxzone_param_8,
	.param .f32 shiftxzone_param_9,
	.param .u64 shiftxzone_param_10
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<32>;
	.reg .b64 	%rd<16>;


	ld.param.u64 	%rd3, [shiftxzone_param_0];
	ld.param.u64 	%rd4, [shiftxzone_param_1];
	ld.param.u32 	%r5, [shiftxzone_param_2];
	ld.param.u32 	%r6, [shiftxzone_param_3];
	ld.param.u32 	%r7, [shiftxzone_param_4];
	ld.param.u32 	%r8, [shiftxzone_param_5];
	ld.param.u32 	%r9, [shiftxzone_param_6];
	ld.param.u32 	%r10, [shiftxzone_param_7];
	ld.param.u64 	%rd5, [shiftxzone_param_10];
	cvta.to.global.u64 	%rd1, %rd5;
	mov.u32 	%r11, %ntid.x;
	mov.u32 	%r12, %ctaid.x;
	mov.u32 	%r13, %tid.x;
	mad.lo.s32 	%r1, %r11, %r12, %r13;
	mov.u32 	%r14, %ntid.y;
	mov.u32 	%r15, %ctaid.y;
	mov.u32 	%r16, %tid.y;
	mad.lo.s32 	%r2, %r14, %r15, %r16;
	mov.u32 	%r17, %ntid.z;
	mov.u32 	%r18, %ctaid.z;
	mov.u32 	%r19, %tid.z;
	mad.lo.s32 	%r3, %r17, %r18, %r19;
	setp.ge.s32	%p1, %r2, %r6;
	setp.ge.s32	%p2, %r1, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	BB0_4;

	mad.lo.s32 	%r20, %r3, %r6, %r2;
	mad.lo.s32 	%r21, %r20, %r5, %r1;
	add.s32 	%r22, %r8, %r5;
	add.s32 	%r23, %r22, %r1;
	rem.s32 	%r24, %r23, %r5;
	add.s32 	%r25, %r9, %r6;
	add.s32 	%r26, %r25, %r2;
	rem.s32 	%r27, %r26, %r6;
	add.s32 	%r28, %r10, %r7;
	add.s32 	%r29, %r28, %r3;
	rem.s32 	%r30, %r29, %r7;
	mad.lo.s32 	%r31, %r30, %r6, %r27;
	mad.lo.s32 	%r4, %r31, %r5, %r24;
	cvt.s64.s32	%rd2, %r21;
	mul.wide.s32 	%rd6, %r21, 4;
	add.s64 	%rd7, %rd1, %rd6;
	ld.global.nc.f32 	%f1, [%rd7];
	setp.neu.f32	%p6, %f1, 0f00000000;
	@%p6 bra 	BB0_3;

	mul.wide.s32 	%rd8, %r4, 4;
	add.s64 	%rd9, %rd1, %rd8;
	ld.global.nc.f32 	%f2, [%rd9];
	setp.eq.f32	%p7, %f2, 0f00000000;
	@%p7 bra 	BB0_4;

BB0_3:
	cvta.to.global.u64 	%rd10, %rd3;
	cvta.to.global.u64 	%rd11, %rd4;
	shl.b64 	%rd12, %rd2, 2;
	add.s64 	%rd13, %rd11, %rd12;
	ld.global.nc.f32 	%f3, [%rd13];
	mul.wide.s32 	%rd14, %r4, 4;
	add.s64 	%rd15, %rd10, %rd14;
	st.global.f32 	[%rd15], %f3;

BB0_4:
	ret;
}


`
   shiftxzone_ptx_50 = `
.version 5.0
.target sm_50
.address_size 64

	// .globl	shiftxzone

.visible .entry shiftxzone(
	.param .u64 shiftxzone_param_0,
	.param .u64 shiftxzone_param_1,
	.param .u32 shiftxzone_param_2,
	.param .u32 shiftxzone_param_3,
	.param .u32 shiftxzone_param_4,
	.param .u32 shiftxzone_param_5,
	.param .u32 shiftxzone_param_6,
	.param .u32 shiftxzone_param_7,
	.param .f32 shiftxzone_param_8,
	.param .f32 shiftxzone_param_9,
	.param .u64 shiftxzone_param_10
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<32>;
	.reg .b64 	%rd<16>;


	ld.param.u64 	%rd3, [shiftxzone_param_0];
	ld.param.u64 	%rd4, [shiftxzone_param_1];
	ld.param.u32 	%r5, [shiftxzone_param_2];
	ld.param.u32 	%r6, [shiftxzone_param_3];
	ld.param.u32 	%r7, [shiftxzone_param_4];
	ld.param.u32 	%r8, [shiftxzone_param_5];
	ld.param.u32 	%r9, [shiftxzone_param_6];
	ld.param.u32 	%r10, [shiftxzone_param_7];
	ld.param.u64 	%rd5, [shiftxzone_param_10];
	cvta.to.global.u64 	%rd1, %rd5;
	mov.u32 	%r11, %ntid.x;
	mov.u32 	%r12, %ctaid.x;
	mov.u32 	%r13, %tid.x;
	mad.lo.s32 	%r1, %r11, %r12, %r13;
	mov.u32 	%r14, %ntid.y;
	mov.u32 	%r15, %ctaid.y;
	mov.u32 	%r16, %tid.y;
	mad.lo.s32 	%r2, %r14, %r15, %r16;
	mov.u32 	%r17, %ntid.z;
	mov.u32 	%r18, %ctaid.z;
	mov.u32 	%r19, %tid.z;
	mad.lo.s32 	%r3, %r17, %r18, %r19;
	setp.ge.s32	%p1, %r2, %r6;
	setp.ge.s32	%p2, %r1, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	BB0_4;

	mad.lo.s32 	%r20, %r3, %r6, %r2;
	mad.lo.s32 	%r21, %r20, %r5, %r1;
	add.s32 	%r22, %r8, %r5;
	add.s32 	%r23, %r22, %r1;
	rem.s32 	%r24, %r23, %r5;
	add.s32 	%r25, %r9, %r6;
	add.s32 	%r26, %r25, %r2;
	rem.s32 	%r27, %r26, %r6;
	add.s32 	%r28, %r10, %r7;
	add.s32 	%r29, %r28, %r3;
	rem.s32 	%r30, %r29, %r7;
	mad.lo.s32 	%r31, %r30, %r6, %r27;
	mad.lo.s32 	%r4, %r31, %r5, %r24;
	cvt.s64.s32	%rd2, %r21;
	mul.wide.s32 	%rd6, %r21, 4;
	add.s64 	%rd7, %rd1, %rd6;
	ld.global.nc.f32 	%f1, [%rd7];
	setp.neu.f32	%p6, %f1, 0f00000000;
	@%p6 bra 	BB0_3;

	mul.wide.s32 	%rd8, %r4, 4;
	add.s64 	%rd9, %rd1, %rd8;
	ld.global.nc.f32 	%f2, [%rd9];
	setp.eq.f32	%p7, %f2, 0f00000000;
	@%p7 bra 	BB0_4;

BB0_3:
	cvta.to.global.u64 	%rd10, %rd3;
	cvta.to.global.u64 	%rd11, %rd4;
	shl.b64 	%rd12, %rd2, 2;
	add.s64 	%rd13, %rd11, %rd12;
	ld.global.nc.f32 	%f3, [%rd13];
	mul.wide.s32 	%rd14, %r4, 4;
	add.s64 	%rd15, %rd10, %rd14;
	st.global.f32 	[%rd15], %f3;

BB0_4:
	ret;
}


`
   shiftxzone_ptx_52 = `
.version 5.0
.target sm_52
.address_size 64

	// .globl	shiftxzone

.visible .entry shiftxzone(
	.param .u64 shiftxzone_param_0,
	.param .u64 shiftxzone_param_1,
	.param .u32 shiftxzone_param_2,
	.param .u32 shiftxzone_param_3,
	.param .u32 shiftxzone_param_4,
	.param .u32 shiftxzone_param_5,
	.param .u32 shiftxzone_param_6,
	.param .u32 shiftxzone_param_7,
	.param .f32 shiftxzone_param_8,
	.param .f32 shiftxzone_param_9,
	.param .u64 shiftxzone_param_10
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<32>;
	.reg .b64 	%rd<16>;


	ld.param.u64 	%rd3, [shiftxzone_param_0];
	ld.param.u64 	%rd4, [shiftxzone_param_1];
	ld.param.u32 	%r5, [shiftxzone_param_2];
	ld.param.u32 	%r6, [shiftxzone_param_3];
	ld.param.u32 	%r7, [shiftxzone_param_4];
	ld.param.u32 	%r8, [shiftxzone_param_5];
	ld.param.u32 	%r9, [shiftxzone_param_6];
	ld.param.u32 	%r10, [shiftxzone_param_7];
	ld.param.u64 	%rd5, [shiftxzone_param_10];
	cvta.to.global.u64 	%rd1, %rd5;
	mov.u32 	%r11, %ntid.x;
	mov.u32 	%r12, %ctaid.x;
	mov.u32 	%r13, %tid.x;
	mad.lo.s32 	%r1, %r11, %r12, %r13;
	mov.u32 	%r14, %ntid.y;
	mov.u32 	%r15, %ctaid.y;
	mov.u32 	%r16, %tid.y;
	mad.lo.s32 	%r2, %r14, %r15, %r16;
	mov.u32 	%r17, %ntid.z;
	mov.u32 	%r18, %ctaid.z;
	mov.u32 	%r19, %tid.z;
	mad.lo.s32 	%r3, %r17, %r18, %r19;
	setp.ge.s32	%p1, %r2, %r6;
	setp.ge.s32	%p2, %r1, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	BB0_4;

	mad.lo.s32 	%r20, %r3, %r6, %r2;
	mad.lo.s32 	%r21, %r20, %r5, %r1;
	add.s32 	%r22, %r8, %r5;
	add.s32 	%r23, %r22, %r1;
	rem.s32 	%r24, %r23, %r5;
	add.s32 	%r25, %r9, %r6;
	add.s32 	%r26, %r25, %r2;
	rem.s32 	%r27, %r26, %r6;
	add.s32 	%r28, %r10, %r7;
	add.s32 	%r29, %r28, %r3;
	rem.s32 	%r30, %r29, %r7;
	mad.lo.s32 	%r31, %r30, %r6, %r27;
	mad.lo.s32 	%r4, %r31, %r5, %r24;
	cvt.s64.s32	%rd2, %r21;
	mul.wide.s32 	%rd6, %r21, 4;
	add.s64 	%rd7, %rd1, %rd6;
	ld.global.nc.f32 	%f1, [%rd7];
	setp.neu.f32	%p6, %f1, 0f00000000;
	@%p6 bra 	BB0_3;

	mul.wide.s32 	%rd8, %r4, 4;
	add.s64 	%rd9, %rd1, %rd8;
	ld.global.nc.f32 	%f2, [%rd9];
	setp.eq.f32	%p7, %f2, 0f00000000;
	@%p7 bra 	BB0_4;

BB0_3:
	cvta.to.global.u64 	%rd10, %rd3;
	cvta.to.global.u64 	%rd11, %rd4;
	shl.b64 	%rd12, %rd2, 2;
	add.s64 	%rd13, %rd11, %rd12;
	ld.global.nc.f32 	%f3, [%rd13];
	mul.wide.s32 	%rd14, %r4, 4;
	add.s64 	%rd15, %rd10, %rd14;
	st.global.f32 	[%rd15], %f3;

BB0_4:
	ret;
}


`
   shiftxzone_ptx_53 = `
.version 5.0
.target sm_53
.address_size 64

	// .globl	shiftxzone

.visible .entry shiftxzone(
	.param .u64 shiftxzone_param_0,
	.param .u64 shiftxzone_param_1,
	.param .u32 shiftxzone_param_2,
	.param .u32 shiftxzone_param_3,
	.param .u32 shiftxzone_param_4,
	.param .u32 shiftxzone_param_5,
	.param .u32 shiftxzone_param_6,
	.param .u32 shiftxzone_param_7,
	.param .f32 shiftxzone_param_8,
	.param .f32 shiftxzone_param_9,
	.param .u64 shiftxzone_param_10
)
{
	.reg .pred 	%p<8>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<32>;
	.reg .b64 	%rd<16>;


	ld.param.u64 	%rd3, [shiftxzone_param_0];
	ld.param.u64 	%rd4, [shiftxzone_param_1];
	ld.param.u32 	%r5, [shiftxzone_param_2];
	ld.param.u32 	%r6, [shiftxzone_param_3];
	ld.param.u32 	%r7, [shiftxzone_param_4];
	ld.param.u32 	%r8, [shiftxzone_param_5];
	ld.param.u32 	%r9, [shiftxzone_param_6];
	ld.param.u32 	%r10, [shiftxzone_param_7];
	ld.param.u64 	%rd5, [shiftxzone_param_10];
	cvta.to.global.u64 	%rd1, %rd5;
	mov.u32 	%r11, %ntid.x;
	mov.u32 	%r12, %ctaid.x;
	mov.u32 	%r13, %tid.x;
	mad.lo.s32 	%r1, %r11, %r12, %r13;
	mov.u32 	%r14, %ntid.y;
	mov.u32 	%r15, %ctaid.y;
	mov.u32 	%r16, %tid.y;
	mad.lo.s32 	%r2, %r14, %r15, %r16;
	mov.u32 	%r17, %ntid.z;
	mov.u32 	%r18, %ctaid.z;
	mov.u32 	%r19, %tid.z;
	mad.lo.s32 	%r3, %r17, %r18, %r19;
	setp.ge.s32	%p1, %r2, %r6;
	setp.ge.s32	%p2, %r1, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	BB0_4;

	mad.lo.s32 	%r20, %r3, %r6, %r2;
	mad.lo.s32 	%r21, %r20, %r5, %r1;
	add.s32 	%r22, %r8, %r5;
	add.s32 	%r23, %r22, %r1;
	rem.s32 	%r24, %r23, %r5;
	add.s32 	%r25, %r9, %r6;
	add.s32 	%r26, %r25, %r2;
	rem.s32 	%r27, %r26, %r6;
	add.s32 	%r28, %r10, %r7;
	add.s32 	%r29, %r28, %r3;
	rem.s32 	%r30, %r29, %r7;
	mad.lo.s32 	%r31, %r30, %r6, %r27;
	mad.lo.s32 	%r4, %r31, %r5, %r24;
	cvt.s64.s32	%rd2, %r21;
	mul.wide.s32 	%rd6, %r21, 4;
	add.s64 	%rd7, %rd1, %rd6;
	ld.global.nc.f32 	%f1, [%rd7];
	setp.neu.f32	%p6, %f1, 0f00000000;
	@%p6 bra 	BB0_3;

	mul.wide.s32 	%rd8, %r4, 4;
	add.s64 	%rd9, %rd1, %rd8;
	ld.global.nc.f32 	%f2, [%rd9];
	setp.eq.f32	%p7, %f2, 0f00000000;
	@%p7 bra 	BB0_4;

BB0_3:
	cvta.to.global.u64 	%rd10, %rd3;
	cvta.to.global.u64 	%rd11, %rd4;
	shl.b64 	%rd12, %rd2, 2;
	add.s64 	%rd13, %rd11, %rd12;
	ld.global.nc.f32 	%f3, [%rd13];
	mul.wide.s32 	%rd14, %r4, 4;
	add.s64 	%rd15, %rd10, %rd14;
	st.global.f32 	[%rd15], %f3;

BB0_4:
	ret;
}


`
 )

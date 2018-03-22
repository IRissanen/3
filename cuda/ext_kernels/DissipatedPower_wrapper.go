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

// CUDA handle for DissipatedPower kernel
var DissipatedPower_code cu.Function

// Stores the arguments for DissipatedPower kernel invocation
type DissipatedPower_args_t struct{
	 arg_powers unsafe.Pointer
	 arg_Bx unsafe.Pointer
	 arg_By unsafe.Pointer
	 arg_Bz unsafe.Pointer
	 arg_Mx unsafe.Pointer
	 arg_My unsafe.Pointer
	 arg_Mz unsafe.Pointer
	 arg_alphas unsafe.Pointer
	 arg_alpha float32
	 arg_msats unsafe.Pointer
	 arg_msat float32
	 arg_volume float32
	 arg_gamma float32
	 arg_Nx int
	 arg_Ny int
	 arg_Nz int
	 argptr [16]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for DissipatedPower kernel invocation
var DissipatedPower_args DissipatedPower_args_t

func init(){
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	 DissipatedPower_args.argptr[0] = unsafe.Pointer(&DissipatedPower_args.arg_powers)
	 DissipatedPower_args.argptr[1] = unsafe.Pointer(&DissipatedPower_args.arg_Bx)
	 DissipatedPower_args.argptr[2] = unsafe.Pointer(&DissipatedPower_args.arg_By)
	 DissipatedPower_args.argptr[3] = unsafe.Pointer(&DissipatedPower_args.arg_Bz)
	 DissipatedPower_args.argptr[4] = unsafe.Pointer(&DissipatedPower_args.arg_Mx)
	 DissipatedPower_args.argptr[5] = unsafe.Pointer(&DissipatedPower_args.arg_My)
	 DissipatedPower_args.argptr[6] = unsafe.Pointer(&DissipatedPower_args.arg_Mz)
	 DissipatedPower_args.argptr[7] = unsafe.Pointer(&DissipatedPower_args.arg_alphas)
	 DissipatedPower_args.argptr[8] = unsafe.Pointer(&DissipatedPower_args.arg_alpha)
	 DissipatedPower_args.argptr[9] = unsafe.Pointer(&DissipatedPower_args.arg_msats)
	 DissipatedPower_args.argptr[10] = unsafe.Pointer(&DissipatedPower_args.arg_msat)
	 DissipatedPower_args.argptr[11] = unsafe.Pointer(&DissipatedPower_args.arg_volume)
	 DissipatedPower_args.argptr[12] = unsafe.Pointer(&DissipatedPower_args.arg_gamma)
	 DissipatedPower_args.argptr[13] = unsafe.Pointer(&DissipatedPower_args.arg_Nx)
	 DissipatedPower_args.argptr[14] = unsafe.Pointer(&DissipatedPower_args.arg_Ny)
	 DissipatedPower_args.argptr[15] = unsafe.Pointer(&DissipatedPower_args.arg_Nz)
	 }

// Wrapper for DissipatedPower CUDA kernel, asynchronous.
func K_ext_DissipatedPower_async ( powers unsafe.Pointer, Bx unsafe.Pointer, By unsafe.Pointer, Bz unsafe.Pointer, Mx unsafe.Pointer, My unsafe.Pointer, Mz unsafe.Pointer, alphas unsafe.Pointer, alpha float32, msats unsafe.Pointer, msat float32, volume float32, gamma float32, Nx int, Ny int, Nz int,  cfg *config) {
	if Synchronous{ // debug
		Sync()
		timer.Start("DissipatedPower")
	}

	DissipatedPower_args.Lock()
	defer DissipatedPower_args.Unlock()

	if DissipatedPower_code == 0{
		DissipatedPower_code = fatbinLoad(DissipatedPower_map, "DissipatedPower")
	}

	 DissipatedPower_args.arg_powers = powers
	 DissipatedPower_args.arg_Bx = Bx
	 DissipatedPower_args.arg_By = By
	 DissipatedPower_args.arg_Bz = Bz
	 DissipatedPower_args.arg_Mx = Mx
	 DissipatedPower_args.arg_My = My
	 DissipatedPower_args.arg_Mz = Mz
	 DissipatedPower_args.arg_alphas = alphas
	 DissipatedPower_args.arg_alpha = alpha
	 DissipatedPower_args.arg_msats = msats
	 DissipatedPower_args.arg_msat = msat
	 DissipatedPower_args.arg_volume = volume
	 DissipatedPower_args.arg_gamma = gamma
	 DissipatedPower_args.arg_Nx = Nx
	 DissipatedPower_args.arg_Ny = Ny
	 DissipatedPower_args.arg_Nz = Nz
	

	args := DissipatedPower_args.argptr[:]
	cu.LaunchKernel(DissipatedPower_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous{ // debug
		Sync()
		timer.Stop("DissipatedPower")
	}
}

// maps compute capability on PTX code for DissipatedPower kernel.
var DissipatedPower_map = map[int]string{ 0: "" ,
20: DissipatedPower_ptx_20 ,
30: DissipatedPower_ptx_30 ,
35: DissipatedPower_ptx_35 ,
50: DissipatedPower_ptx_50 ,
52: DissipatedPower_ptx_52 ,
53: DissipatedPower_ptx_53  }

// DissipatedPower PTX code for various compute capabilities.
const(
  DissipatedPower_ptx_20 = `
.version 5.0
.target sm_20
.address_size 64

	// .globl	DissipatedPower

.visible .entry DissipatedPower(
	.param .u64 DissipatedPower_param_0,
	.param .u64 DissipatedPower_param_1,
	.param .u64 DissipatedPower_param_2,
	.param .u64 DissipatedPower_param_3,
	.param .u64 DissipatedPower_param_4,
	.param .u64 DissipatedPower_param_5,
	.param .u64 DissipatedPower_param_6,
	.param .u64 DissipatedPower_param_7,
	.param .f32 DissipatedPower_param_8,
	.param .u64 DissipatedPower_param_9,
	.param .f32 DissipatedPower_param_10,
	.param .f32 DissipatedPower_param_11,
	.param .f32 DissipatedPower_param_12,
	.param .u32 DissipatedPower_param_13,
	.param .u32 DissipatedPower_param_14,
	.param .u32 DissipatedPower_param_15
)
{
	.reg .pred 	%p<9>;
	.reg .f32 	%f<37>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<31>;


	ld.param.u64 	%rd1, [DissipatedPower_param_0];
	ld.param.u64 	%rd2, [DissipatedPower_param_1];
	ld.param.u64 	%rd3, [DissipatedPower_param_2];
	ld.param.u64 	%rd4, [DissipatedPower_param_3];
	ld.param.u64 	%rd5, [DissipatedPower_param_4];
	ld.param.u64 	%rd6, [DissipatedPower_param_5];
	ld.param.u64 	%rd7, [DissipatedPower_param_6];
	ld.param.u64 	%rd8, [DissipatedPower_param_7];
	ld.param.f32 	%f35, [DissipatedPower_param_8];
	ld.param.u64 	%rd9, [DissipatedPower_param_9];
	ld.param.f32 	%f36, [DissipatedPower_param_10];
	ld.param.f32 	%f7, [DissipatedPower_param_11];
	ld.param.f32 	%f8, [DissipatedPower_param_12];
	ld.param.u32 	%r5, [DissipatedPower_param_13];
	ld.param.u32 	%r6, [DissipatedPower_param_14];
	ld.param.u32 	%r7, [DissipatedPower_param_15];
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r8, %r9, %r10;
	mov.u32 	%r11, %ntid.y;
	mov.u32 	%r12, %ctaid.y;
	mov.u32 	%r13, %tid.y;
	mad.lo.s32 	%r2, %r11, %r12, %r13;
	mov.u32 	%r14, %ntid.z;
	mov.u32 	%r15, %ctaid.z;
	mov.u32 	%r16, %tid.z;
	mad.lo.s32 	%r3, %r14, %r15, %r16;
	setp.ge.s32	%p1, %r2, %r6;
	setp.ge.s32	%p2, %r1, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	BB0_7;

	mad.lo.s32 	%r17, %r3, %r6, %r2;
	mad.lo.s32 	%r4, %r17, %r5, %r1;
	setp.eq.s64	%p6, %rd8, 0;
	@%p6 bra 	BB0_3;

	cvta.to.global.u64 	%rd10, %rd8;
	mul.wide.s32 	%rd11, %r4, 4;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.f32 	%f9, [%rd12];
	mul.f32 	%f35, %f9, %f35;

BB0_3:
	setp.eq.s64	%p7, %rd9, 0;
	@%p7 bra 	BB0_5;

	cvta.to.global.u64 	%rd13, %rd9;
	mul.wide.s32 	%rd14, %r4, 4;
	add.s64 	%rd15, %rd13, %rd14;
	ld.global.f32 	%f10, [%rd15];
	mul.f32 	%f36, %f10, %f36;

BB0_5:
	setp.eq.f32	%p8, %f36, 0f00000000;
	@%p8 bra 	BB0_7;

	cvta.to.global.u64 	%rd16, %rd1;
	cvta.to.global.u64 	%rd17, %rd2;
	cvta.to.global.u64 	%rd18, %rd5;
	cvta.to.global.u64 	%rd19, %rd3;
	cvta.to.global.u64 	%rd20, %rd7;
	cvta.to.global.u64 	%rd21, %rd4;
	cvta.to.global.u64 	%rd22, %rd6;
	mul.wide.s32 	%rd23, %r4, 4;
	add.s64 	%rd24, %rd22, %rd23;
	add.s64 	%rd25, %rd21, %rd23;
	ld.global.f32 	%f11, [%rd25];
	ld.global.f32 	%f12, [%rd24];
	mul.f32 	%f13, %f12, %f11;
	add.s64 	%rd26, %rd20, %rd23;
	add.s64 	%rd27, %rd19, %rd23;
	ld.global.f32 	%f14, [%rd27];
	ld.global.f32 	%f15, [%rd26];
	mul.f32 	%f16, %f15, %f14;
	sub.f32 	%f17, %f13, %f16;
	add.s64 	%rd28, %rd18, %rd23;
	ld.global.f32 	%f18, [%rd28];
	mul.f32 	%f19, %f18, %f11;
	add.s64 	%rd29, %rd17, %rd23;
	ld.global.f32 	%f20, [%rd29];
	mul.f32 	%f21, %f15, %f20;
	sub.f32 	%f22, %f19, %f21;
	mul.f32 	%f23, %f18, %f14;
	mul.f32 	%f24, %f12, %f20;
	sub.f32 	%f25, %f23, %f24;
	mul.f32 	%f26, %f36, %f7;
	mul.f32 	%f27, %f26, %f8;
	mul.f32 	%f28, %f35, %f27;
	fma.rn.f32 	%f29, %f35, %f35, 0f3F800000;
	div.rn.f32 	%f30, %f28, %f29;
	mul.f32 	%f31, %f22, %f22;
	fma.rn.f32 	%f32, %f17, %f17, %f31;
	fma.rn.f32 	%f33, %f25, %f25, %f32;
	mul.f32 	%f34, %f30, %f33;
	add.s64 	%rd30, %rd16, %rd23;
	st.global.f32 	[%rd30], %f34;

BB0_7:
	ret;
}


`
   DissipatedPower_ptx_30 = `
.version 5.0
.target sm_30
.address_size 64

	// .globl	DissipatedPower

.visible .entry DissipatedPower(
	.param .u64 DissipatedPower_param_0,
	.param .u64 DissipatedPower_param_1,
	.param .u64 DissipatedPower_param_2,
	.param .u64 DissipatedPower_param_3,
	.param .u64 DissipatedPower_param_4,
	.param .u64 DissipatedPower_param_5,
	.param .u64 DissipatedPower_param_6,
	.param .u64 DissipatedPower_param_7,
	.param .f32 DissipatedPower_param_8,
	.param .u64 DissipatedPower_param_9,
	.param .f32 DissipatedPower_param_10,
	.param .f32 DissipatedPower_param_11,
	.param .f32 DissipatedPower_param_12,
	.param .u32 DissipatedPower_param_13,
	.param .u32 DissipatedPower_param_14,
	.param .u32 DissipatedPower_param_15
)
{
	.reg .pred 	%p<9>;
	.reg .f32 	%f<37>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<31>;


	ld.param.u64 	%rd1, [DissipatedPower_param_0];
	ld.param.u64 	%rd2, [DissipatedPower_param_1];
	ld.param.u64 	%rd3, [DissipatedPower_param_2];
	ld.param.u64 	%rd4, [DissipatedPower_param_3];
	ld.param.u64 	%rd5, [DissipatedPower_param_4];
	ld.param.u64 	%rd6, [DissipatedPower_param_5];
	ld.param.u64 	%rd7, [DissipatedPower_param_6];
	ld.param.u64 	%rd8, [DissipatedPower_param_7];
	ld.param.f32 	%f35, [DissipatedPower_param_8];
	ld.param.u64 	%rd9, [DissipatedPower_param_9];
	ld.param.f32 	%f36, [DissipatedPower_param_10];
	ld.param.f32 	%f7, [DissipatedPower_param_11];
	ld.param.f32 	%f8, [DissipatedPower_param_12];
	ld.param.u32 	%r5, [DissipatedPower_param_13];
	ld.param.u32 	%r6, [DissipatedPower_param_14];
	ld.param.u32 	%r7, [DissipatedPower_param_15];
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r8, %r9, %r10;
	mov.u32 	%r11, %ntid.y;
	mov.u32 	%r12, %ctaid.y;
	mov.u32 	%r13, %tid.y;
	mad.lo.s32 	%r2, %r11, %r12, %r13;
	mov.u32 	%r14, %ntid.z;
	mov.u32 	%r15, %ctaid.z;
	mov.u32 	%r16, %tid.z;
	mad.lo.s32 	%r3, %r14, %r15, %r16;
	setp.ge.s32	%p1, %r2, %r6;
	setp.ge.s32	%p2, %r1, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	BB0_7;

	mad.lo.s32 	%r17, %r3, %r6, %r2;
	mad.lo.s32 	%r4, %r17, %r5, %r1;
	setp.eq.s64	%p6, %rd8, 0;
	@%p6 bra 	BB0_3;

	cvta.to.global.u64 	%rd10, %rd8;
	mul.wide.s32 	%rd11, %r4, 4;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.f32 	%f9, [%rd12];
	mul.f32 	%f35, %f9, %f35;

BB0_3:
	setp.eq.s64	%p7, %rd9, 0;
	@%p7 bra 	BB0_5;

	cvta.to.global.u64 	%rd13, %rd9;
	mul.wide.s32 	%rd14, %r4, 4;
	add.s64 	%rd15, %rd13, %rd14;
	ld.global.f32 	%f10, [%rd15];
	mul.f32 	%f36, %f10, %f36;

BB0_5:
	setp.eq.f32	%p8, %f36, 0f00000000;
	@%p8 bra 	BB0_7;

	cvta.to.global.u64 	%rd16, %rd1;
	cvta.to.global.u64 	%rd17, %rd2;
	cvta.to.global.u64 	%rd18, %rd5;
	cvta.to.global.u64 	%rd19, %rd3;
	cvta.to.global.u64 	%rd20, %rd7;
	cvta.to.global.u64 	%rd21, %rd4;
	cvta.to.global.u64 	%rd22, %rd6;
	mul.wide.s32 	%rd23, %r4, 4;
	add.s64 	%rd24, %rd22, %rd23;
	add.s64 	%rd25, %rd21, %rd23;
	ld.global.f32 	%f11, [%rd25];
	ld.global.f32 	%f12, [%rd24];
	mul.f32 	%f13, %f12, %f11;
	add.s64 	%rd26, %rd20, %rd23;
	add.s64 	%rd27, %rd19, %rd23;
	ld.global.f32 	%f14, [%rd27];
	ld.global.f32 	%f15, [%rd26];
	mul.f32 	%f16, %f15, %f14;
	sub.f32 	%f17, %f13, %f16;
	add.s64 	%rd28, %rd18, %rd23;
	ld.global.f32 	%f18, [%rd28];
	mul.f32 	%f19, %f18, %f11;
	add.s64 	%rd29, %rd17, %rd23;
	ld.global.f32 	%f20, [%rd29];
	mul.f32 	%f21, %f15, %f20;
	sub.f32 	%f22, %f19, %f21;
	mul.f32 	%f23, %f18, %f14;
	mul.f32 	%f24, %f12, %f20;
	sub.f32 	%f25, %f23, %f24;
	mul.f32 	%f26, %f36, %f7;
	mul.f32 	%f27, %f26, %f8;
	mul.f32 	%f28, %f35, %f27;
	fma.rn.f32 	%f29, %f35, %f35, 0f3F800000;
	div.rn.f32 	%f30, %f28, %f29;
	mul.f32 	%f31, %f22, %f22;
	fma.rn.f32 	%f32, %f17, %f17, %f31;
	fma.rn.f32 	%f33, %f25, %f25, %f32;
	mul.f32 	%f34, %f30, %f33;
	add.s64 	%rd30, %rd16, %rd23;
	st.global.f32 	[%rd30], %f34;

BB0_7:
	ret;
}


`
   DissipatedPower_ptx_35 = `
.version 5.0
.target sm_35
.address_size 64

	// .globl	DissipatedPower

.visible .entry DissipatedPower(
	.param .u64 DissipatedPower_param_0,
	.param .u64 DissipatedPower_param_1,
	.param .u64 DissipatedPower_param_2,
	.param .u64 DissipatedPower_param_3,
	.param .u64 DissipatedPower_param_4,
	.param .u64 DissipatedPower_param_5,
	.param .u64 DissipatedPower_param_6,
	.param .u64 DissipatedPower_param_7,
	.param .f32 DissipatedPower_param_8,
	.param .u64 DissipatedPower_param_9,
	.param .f32 DissipatedPower_param_10,
	.param .f32 DissipatedPower_param_11,
	.param .f32 DissipatedPower_param_12,
	.param .u32 DissipatedPower_param_13,
	.param .u32 DissipatedPower_param_14,
	.param .u32 DissipatedPower_param_15
)
{
	.reg .pred 	%p<9>;
	.reg .f32 	%f<37>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<31>;


	ld.param.u64 	%rd1, [DissipatedPower_param_0];
	ld.param.u64 	%rd2, [DissipatedPower_param_1];
	ld.param.u64 	%rd3, [DissipatedPower_param_2];
	ld.param.u64 	%rd4, [DissipatedPower_param_3];
	ld.param.u64 	%rd5, [DissipatedPower_param_4];
	ld.param.u64 	%rd6, [DissipatedPower_param_5];
	ld.param.u64 	%rd7, [DissipatedPower_param_6];
	ld.param.u64 	%rd8, [DissipatedPower_param_7];
	ld.param.f32 	%f35, [DissipatedPower_param_8];
	ld.param.u64 	%rd9, [DissipatedPower_param_9];
	ld.param.f32 	%f36, [DissipatedPower_param_10];
	ld.param.f32 	%f7, [DissipatedPower_param_11];
	ld.param.f32 	%f8, [DissipatedPower_param_12];
	ld.param.u32 	%r5, [DissipatedPower_param_13];
	ld.param.u32 	%r6, [DissipatedPower_param_14];
	ld.param.u32 	%r7, [DissipatedPower_param_15];
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r8, %r9, %r10;
	mov.u32 	%r11, %ntid.y;
	mov.u32 	%r12, %ctaid.y;
	mov.u32 	%r13, %tid.y;
	mad.lo.s32 	%r2, %r11, %r12, %r13;
	mov.u32 	%r14, %ntid.z;
	mov.u32 	%r15, %ctaid.z;
	mov.u32 	%r16, %tid.z;
	mad.lo.s32 	%r3, %r14, %r15, %r16;
	setp.ge.s32	%p1, %r2, %r6;
	setp.ge.s32	%p2, %r1, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	BB0_7;

	mad.lo.s32 	%r17, %r3, %r6, %r2;
	mad.lo.s32 	%r4, %r17, %r5, %r1;
	setp.eq.s64	%p6, %rd8, 0;
	@%p6 bra 	BB0_3;

	cvta.to.global.u64 	%rd10, %rd8;
	mul.wide.s32 	%rd11, %r4, 4;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.nc.f32 	%f9, [%rd12];
	mul.f32 	%f35, %f9, %f35;

BB0_3:
	setp.eq.s64	%p7, %rd9, 0;
	@%p7 bra 	BB0_5;

	cvta.to.global.u64 	%rd13, %rd9;
	mul.wide.s32 	%rd14, %r4, 4;
	add.s64 	%rd15, %rd13, %rd14;
	ld.global.nc.f32 	%f10, [%rd15];
	mul.f32 	%f36, %f10, %f36;

BB0_5:
	setp.eq.f32	%p8, %f36, 0f00000000;
	@%p8 bra 	BB0_7;

	cvta.to.global.u64 	%rd16, %rd1;
	cvta.to.global.u64 	%rd17, %rd2;
	cvta.to.global.u64 	%rd18, %rd5;
	cvta.to.global.u64 	%rd19, %rd3;
	cvta.to.global.u64 	%rd20, %rd7;
	cvta.to.global.u64 	%rd21, %rd4;
	cvta.to.global.u64 	%rd22, %rd6;
	mul.wide.s32 	%rd23, %r4, 4;
	add.s64 	%rd24, %rd22, %rd23;
	add.s64 	%rd25, %rd21, %rd23;
	ld.global.nc.f32 	%f11, [%rd25];
	ld.global.nc.f32 	%f12, [%rd24];
	mul.f32 	%f13, %f12, %f11;
	add.s64 	%rd26, %rd20, %rd23;
	add.s64 	%rd27, %rd19, %rd23;
	ld.global.nc.f32 	%f14, [%rd27];
	ld.global.nc.f32 	%f15, [%rd26];
	mul.f32 	%f16, %f15, %f14;
	sub.f32 	%f17, %f13, %f16;
	add.s64 	%rd28, %rd18, %rd23;
	ld.global.nc.f32 	%f18, [%rd28];
	mul.f32 	%f19, %f18, %f11;
	add.s64 	%rd29, %rd17, %rd23;
	ld.global.nc.f32 	%f20, [%rd29];
	mul.f32 	%f21, %f15, %f20;
	sub.f32 	%f22, %f19, %f21;
	mul.f32 	%f23, %f18, %f14;
	mul.f32 	%f24, %f12, %f20;
	sub.f32 	%f25, %f23, %f24;
	mul.f32 	%f26, %f36, %f7;
	mul.f32 	%f27, %f26, %f8;
	mul.f32 	%f28, %f35, %f27;
	fma.rn.f32 	%f29, %f35, %f35, 0f3F800000;
	div.rn.f32 	%f30, %f28, %f29;
	mul.f32 	%f31, %f22, %f22;
	fma.rn.f32 	%f32, %f17, %f17, %f31;
	fma.rn.f32 	%f33, %f25, %f25, %f32;
	mul.f32 	%f34, %f30, %f33;
	add.s64 	%rd30, %rd16, %rd23;
	st.global.f32 	[%rd30], %f34;

BB0_7:
	ret;
}


`
   DissipatedPower_ptx_50 = `
.version 5.0
.target sm_50
.address_size 64

	// .globl	DissipatedPower

.visible .entry DissipatedPower(
	.param .u64 DissipatedPower_param_0,
	.param .u64 DissipatedPower_param_1,
	.param .u64 DissipatedPower_param_2,
	.param .u64 DissipatedPower_param_3,
	.param .u64 DissipatedPower_param_4,
	.param .u64 DissipatedPower_param_5,
	.param .u64 DissipatedPower_param_6,
	.param .u64 DissipatedPower_param_7,
	.param .f32 DissipatedPower_param_8,
	.param .u64 DissipatedPower_param_9,
	.param .f32 DissipatedPower_param_10,
	.param .f32 DissipatedPower_param_11,
	.param .f32 DissipatedPower_param_12,
	.param .u32 DissipatedPower_param_13,
	.param .u32 DissipatedPower_param_14,
	.param .u32 DissipatedPower_param_15
)
{
	.reg .pred 	%p<9>;
	.reg .f32 	%f<37>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<31>;


	ld.param.u64 	%rd1, [DissipatedPower_param_0];
	ld.param.u64 	%rd2, [DissipatedPower_param_1];
	ld.param.u64 	%rd3, [DissipatedPower_param_2];
	ld.param.u64 	%rd4, [DissipatedPower_param_3];
	ld.param.u64 	%rd5, [DissipatedPower_param_4];
	ld.param.u64 	%rd6, [DissipatedPower_param_5];
	ld.param.u64 	%rd7, [DissipatedPower_param_6];
	ld.param.u64 	%rd8, [DissipatedPower_param_7];
	ld.param.f32 	%f35, [DissipatedPower_param_8];
	ld.param.u64 	%rd9, [DissipatedPower_param_9];
	ld.param.f32 	%f36, [DissipatedPower_param_10];
	ld.param.f32 	%f7, [DissipatedPower_param_11];
	ld.param.f32 	%f8, [DissipatedPower_param_12];
	ld.param.u32 	%r5, [DissipatedPower_param_13];
	ld.param.u32 	%r6, [DissipatedPower_param_14];
	ld.param.u32 	%r7, [DissipatedPower_param_15];
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r8, %r9, %r10;
	mov.u32 	%r11, %ntid.y;
	mov.u32 	%r12, %ctaid.y;
	mov.u32 	%r13, %tid.y;
	mad.lo.s32 	%r2, %r11, %r12, %r13;
	mov.u32 	%r14, %ntid.z;
	mov.u32 	%r15, %ctaid.z;
	mov.u32 	%r16, %tid.z;
	mad.lo.s32 	%r3, %r14, %r15, %r16;
	setp.ge.s32	%p1, %r2, %r6;
	setp.ge.s32	%p2, %r1, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	BB0_7;

	mad.lo.s32 	%r17, %r3, %r6, %r2;
	mad.lo.s32 	%r4, %r17, %r5, %r1;
	setp.eq.s64	%p6, %rd8, 0;
	@%p6 bra 	BB0_3;

	cvta.to.global.u64 	%rd10, %rd8;
	mul.wide.s32 	%rd11, %r4, 4;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.nc.f32 	%f9, [%rd12];
	mul.f32 	%f35, %f9, %f35;

BB0_3:
	setp.eq.s64	%p7, %rd9, 0;
	@%p7 bra 	BB0_5;

	cvta.to.global.u64 	%rd13, %rd9;
	mul.wide.s32 	%rd14, %r4, 4;
	add.s64 	%rd15, %rd13, %rd14;
	ld.global.nc.f32 	%f10, [%rd15];
	mul.f32 	%f36, %f10, %f36;

BB0_5:
	setp.eq.f32	%p8, %f36, 0f00000000;
	@%p8 bra 	BB0_7;

	cvta.to.global.u64 	%rd16, %rd1;
	cvta.to.global.u64 	%rd17, %rd2;
	cvta.to.global.u64 	%rd18, %rd5;
	cvta.to.global.u64 	%rd19, %rd3;
	cvta.to.global.u64 	%rd20, %rd7;
	cvta.to.global.u64 	%rd21, %rd4;
	cvta.to.global.u64 	%rd22, %rd6;
	mul.wide.s32 	%rd23, %r4, 4;
	add.s64 	%rd24, %rd22, %rd23;
	add.s64 	%rd25, %rd21, %rd23;
	ld.global.nc.f32 	%f11, [%rd25];
	ld.global.nc.f32 	%f12, [%rd24];
	mul.f32 	%f13, %f12, %f11;
	add.s64 	%rd26, %rd20, %rd23;
	add.s64 	%rd27, %rd19, %rd23;
	ld.global.nc.f32 	%f14, [%rd27];
	ld.global.nc.f32 	%f15, [%rd26];
	mul.f32 	%f16, %f15, %f14;
	sub.f32 	%f17, %f13, %f16;
	add.s64 	%rd28, %rd18, %rd23;
	ld.global.nc.f32 	%f18, [%rd28];
	mul.f32 	%f19, %f18, %f11;
	add.s64 	%rd29, %rd17, %rd23;
	ld.global.nc.f32 	%f20, [%rd29];
	mul.f32 	%f21, %f15, %f20;
	sub.f32 	%f22, %f19, %f21;
	mul.f32 	%f23, %f18, %f14;
	mul.f32 	%f24, %f12, %f20;
	sub.f32 	%f25, %f23, %f24;
	mul.f32 	%f26, %f36, %f7;
	mul.f32 	%f27, %f26, %f8;
	mul.f32 	%f28, %f35, %f27;
	fma.rn.f32 	%f29, %f35, %f35, 0f3F800000;
	div.rn.f32 	%f30, %f28, %f29;
	mul.f32 	%f31, %f22, %f22;
	fma.rn.f32 	%f32, %f17, %f17, %f31;
	fma.rn.f32 	%f33, %f25, %f25, %f32;
	mul.f32 	%f34, %f30, %f33;
	add.s64 	%rd30, %rd16, %rd23;
	st.global.f32 	[%rd30], %f34;

BB0_7:
	ret;
}


`
   DissipatedPower_ptx_52 = `
.version 5.0
.target sm_52
.address_size 64

	// .globl	DissipatedPower

.visible .entry DissipatedPower(
	.param .u64 DissipatedPower_param_0,
	.param .u64 DissipatedPower_param_1,
	.param .u64 DissipatedPower_param_2,
	.param .u64 DissipatedPower_param_3,
	.param .u64 DissipatedPower_param_4,
	.param .u64 DissipatedPower_param_5,
	.param .u64 DissipatedPower_param_6,
	.param .u64 DissipatedPower_param_7,
	.param .f32 DissipatedPower_param_8,
	.param .u64 DissipatedPower_param_9,
	.param .f32 DissipatedPower_param_10,
	.param .f32 DissipatedPower_param_11,
	.param .f32 DissipatedPower_param_12,
	.param .u32 DissipatedPower_param_13,
	.param .u32 DissipatedPower_param_14,
	.param .u32 DissipatedPower_param_15
)
{
	.reg .pred 	%p<9>;
	.reg .f32 	%f<37>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<31>;


	ld.param.u64 	%rd1, [DissipatedPower_param_0];
	ld.param.u64 	%rd2, [DissipatedPower_param_1];
	ld.param.u64 	%rd3, [DissipatedPower_param_2];
	ld.param.u64 	%rd4, [DissipatedPower_param_3];
	ld.param.u64 	%rd5, [DissipatedPower_param_4];
	ld.param.u64 	%rd6, [DissipatedPower_param_5];
	ld.param.u64 	%rd7, [DissipatedPower_param_6];
	ld.param.u64 	%rd8, [DissipatedPower_param_7];
	ld.param.f32 	%f35, [DissipatedPower_param_8];
	ld.param.u64 	%rd9, [DissipatedPower_param_9];
	ld.param.f32 	%f36, [DissipatedPower_param_10];
	ld.param.f32 	%f7, [DissipatedPower_param_11];
	ld.param.f32 	%f8, [DissipatedPower_param_12];
	ld.param.u32 	%r5, [DissipatedPower_param_13];
	ld.param.u32 	%r6, [DissipatedPower_param_14];
	ld.param.u32 	%r7, [DissipatedPower_param_15];
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r8, %r9, %r10;
	mov.u32 	%r11, %ntid.y;
	mov.u32 	%r12, %ctaid.y;
	mov.u32 	%r13, %tid.y;
	mad.lo.s32 	%r2, %r11, %r12, %r13;
	mov.u32 	%r14, %ntid.z;
	mov.u32 	%r15, %ctaid.z;
	mov.u32 	%r16, %tid.z;
	mad.lo.s32 	%r3, %r14, %r15, %r16;
	setp.ge.s32	%p1, %r2, %r6;
	setp.ge.s32	%p2, %r1, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	BB0_7;

	mad.lo.s32 	%r17, %r3, %r6, %r2;
	mad.lo.s32 	%r4, %r17, %r5, %r1;
	setp.eq.s64	%p6, %rd8, 0;
	@%p6 bra 	BB0_3;

	cvta.to.global.u64 	%rd10, %rd8;
	mul.wide.s32 	%rd11, %r4, 4;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.nc.f32 	%f9, [%rd12];
	mul.f32 	%f35, %f9, %f35;

BB0_3:
	setp.eq.s64	%p7, %rd9, 0;
	@%p7 bra 	BB0_5;

	cvta.to.global.u64 	%rd13, %rd9;
	mul.wide.s32 	%rd14, %r4, 4;
	add.s64 	%rd15, %rd13, %rd14;
	ld.global.nc.f32 	%f10, [%rd15];
	mul.f32 	%f36, %f10, %f36;

BB0_5:
	setp.eq.f32	%p8, %f36, 0f00000000;
	@%p8 bra 	BB0_7;

	cvta.to.global.u64 	%rd16, %rd1;
	cvta.to.global.u64 	%rd17, %rd2;
	cvta.to.global.u64 	%rd18, %rd5;
	cvta.to.global.u64 	%rd19, %rd3;
	cvta.to.global.u64 	%rd20, %rd7;
	cvta.to.global.u64 	%rd21, %rd4;
	cvta.to.global.u64 	%rd22, %rd6;
	mul.wide.s32 	%rd23, %r4, 4;
	add.s64 	%rd24, %rd22, %rd23;
	add.s64 	%rd25, %rd21, %rd23;
	ld.global.nc.f32 	%f11, [%rd25];
	ld.global.nc.f32 	%f12, [%rd24];
	mul.f32 	%f13, %f12, %f11;
	add.s64 	%rd26, %rd20, %rd23;
	add.s64 	%rd27, %rd19, %rd23;
	ld.global.nc.f32 	%f14, [%rd27];
	ld.global.nc.f32 	%f15, [%rd26];
	mul.f32 	%f16, %f15, %f14;
	sub.f32 	%f17, %f13, %f16;
	add.s64 	%rd28, %rd18, %rd23;
	ld.global.nc.f32 	%f18, [%rd28];
	mul.f32 	%f19, %f18, %f11;
	add.s64 	%rd29, %rd17, %rd23;
	ld.global.nc.f32 	%f20, [%rd29];
	mul.f32 	%f21, %f15, %f20;
	sub.f32 	%f22, %f19, %f21;
	mul.f32 	%f23, %f18, %f14;
	mul.f32 	%f24, %f12, %f20;
	sub.f32 	%f25, %f23, %f24;
	mul.f32 	%f26, %f36, %f7;
	mul.f32 	%f27, %f26, %f8;
	mul.f32 	%f28, %f35, %f27;
	fma.rn.f32 	%f29, %f35, %f35, 0f3F800000;
	div.rn.f32 	%f30, %f28, %f29;
	mul.f32 	%f31, %f22, %f22;
	fma.rn.f32 	%f32, %f17, %f17, %f31;
	fma.rn.f32 	%f33, %f25, %f25, %f32;
	mul.f32 	%f34, %f30, %f33;
	add.s64 	%rd30, %rd16, %rd23;
	st.global.f32 	[%rd30], %f34;

BB0_7:
	ret;
}


`
   DissipatedPower_ptx_53 = `
.version 5.0
.target sm_53
.address_size 64

	// .globl	DissipatedPower

.visible .entry DissipatedPower(
	.param .u64 DissipatedPower_param_0,
	.param .u64 DissipatedPower_param_1,
	.param .u64 DissipatedPower_param_2,
	.param .u64 DissipatedPower_param_3,
	.param .u64 DissipatedPower_param_4,
	.param .u64 DissipatedPower_param_5,
	.param .u64 DissipatedPower_param_6,
	.param .u64 DissipatedPower_param_7,
	.param .f32 DissipatedPower_param_8,
	.param .u64 DissipatedPower_param_9,
	.param .f32 DissipatedPower_param_10,
	.param .f32 DissipatedPower_param_11,
	.param .f32 DissipatedPower_param_12,
	.param .u32 DissipatedPower_param_13,
	.param .u32 DissipatedPower_param_14,
	.param .u32 DissipatedPower_param_15
)
{
	.reg .pred 	%p<9>;
	.reg .f32 	%f<37>;
	.reg .b32 	%r<18>;
	.reg .b64 	%rd<31>;


	ld.param.u64 	%rd1, [DissipatedPower_param_0];
	ld.param.u64 	%rd2, [DissipatedPower_param_1];
	ld.param.u64 	%rd3, [DissipatedPower_param_2];
	ld.param.u64 	%rd4, [DissipatedPower_param_3];
	ld.param.u64 	%rd5, [DissipatedPower_param_4];
	ld.param.u64 	%rd6, [DissipatedPower_param_5];
	ld.param.u64 	%rd7, [DissipatedPower_param_6];
	ld.param.u64 	%rd8, [DissipatedPower_param_7];
	ld.param.f32 	%f35, [DissipatedPower_param_8];
	ld.param.u64 	%rd9, [DissipatedPower_param_9];
	ld.param.f32 	%f36, [DissipatedPower_param_10];
	ld.param.f32 	%f7, [DissipatedPower_param_11];
	ld.param.f32 	%f8, [DissipatedPower_param_12];
	ld.param.u32 	%r5, [DissipatedPower_param_13];
	ld.param.u32 	%r6, [DissipatedPower_param_14];
	ld.param.u32 	%r7, [DissipatedPower_param_15];
	mov.u32 	%r8, %ntid.x;
	mov.u32 	%r9, %ctaid.x;
	mov.u32 	%r10, %tid.x;
	mad.lo.s32 	%r1, %r8, %r9, %r10;
	mov.u32 	%r11, %ntid.y;
	mov.u32 	%r12, %ctaid.y;
	mov.u32 	%r13, %tid.y;
	mad.lo.s32 	%r2, %r11, %r12, %r13;
	mov.u32 	%r14, %ntid.z;
	mov.u32 	%r15, %ctaid.z;
	mov.u32 	%r16, %tid.z;
	mad.lo.s32 	%r3, %r14, %r15, %r16;
	setp.ge.s32	%p1, %r2, %r6;
	setp.ge.s32	%p2, %r1, %r5;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32	%p4, %r3, %r7;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	BB0_7;

	mad.lo.s32 	%r17, %r3, %r6, %r2;
	mad.lo.s32 	%r4, %r17, %r5, %r1;
	setp.eq.s64	%p6, %rd8, 0;
	@%p6 bra 	BB0_3;

	cvta.to.global.u64 	%rd10, %rd8;
	mul.wide.s32 	%rd11, %r4, 4;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.nc.f32 	%f9, [%rd12];
	mul.f32 	%f35, %f9, %f35;

BB0_3:
	setp.eq.s64	%p7, %rd9, 0;
	@%p7 bra 	BB0_5;

	cvta.to.global.u64 	%rd13, %rd9;
	mul.wide.s32 	%rd14, %r4, 4;
	add.s64 	%rd15, %rd13, %rd14;
	ld.global.nc.f32 	%f10, [%rd15];
	mul.f32 	%f36, %f10, %f36;

BB0_5:
	setp.eq.f32	%p8, %f36, 0f00000000;
	@%p8 bra 	BB0_7;

	cvta.to.global.u64 	%rd16, %rd1;
	cvta.to.global.u64 	%rd17, %rd2;
	cvta.to.global.u64 	%rd18, %rd5;
	cvta.to.global.u64 	%rd19, %rd3;
	cvta.to.global.u64 	%rd20, %rd7;
	cvta.to.global.u64 	%rd21, %rd4;
	cvta.to.global.u64 	%rd22, %rd6;
	mul.wide.s32 	%rd23, %r4, 4;
	add.s64 	%rd24, %rd22, %rd23;
	add.s64 	%rd25, %rd21, %rd23;
	ld.global.nc.f32 	%f11, [%rd25];
	ld.global.nc.f32 	%f12, [%rd24];
	mul.f32 	%f13, %f12, %f11;
	add.s64 	%rd26, %rd20, %rd23;
	add.s64 	%rd27, %rd19, %rd23;
	ld.global.nc.f32 	%f14, [%rd27];
	ld.global.nc.f32 	%f15, [%rd26];
	mul.f32 	%f16, %f15, %f14;
	sub.f32 	%f17, %f13, %f16;
	add.s64 	%rd28, %rd18, %rd23;
	ld.global.nc.f32 	%f18, [%rd28];
	mul.f32 	%f19, %f18, %f11;
	add.s64 	%rd29, %rd17, %rd23;
	ld.global.nc.f32 	%f20, [%rd29];
	mul.f32 	%f21, %f15, %f20;
	sub.f32 	%f22, %f19, %f21;
	mul.f32 	%f23, %f18, %f14;
	mul.f32 	%f24, %f12, %f20;
	sub.f32 	%f25, %f23, %f24;
	mul.f32 	%f26, %f36, %f7;
	mul.f32 	%f27, %f26, %f8;
	mul.f32 	%f28, %f35, %f27;
	fma.rn.f32 	%f29, %f35, %f35, 0f3F800000;
	div.rn.f32 	%f30, %f28, %f29;
	mul.f32 	%f31, %f22, %f22;
	fma.rn.f32 	%f32, %f17, %f17, %f31;
	fma.rn.f32 	%f33, %f25, %f25, %f32;
	mul.f32 	%f34, %f30, %f33;
	add.s64 	%rd30, %rd16, %rd23;
	st.global.f32 	[%rd30], %f34;

BB0_7:
	ret;
}


`
 )

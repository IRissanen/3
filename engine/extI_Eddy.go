package engine

// Eddy current field

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/mag"
	"github.com/mumax/3/data"
)

var (
	Conductivity  					= 0.0 //TODO: Conductivity as a region specific variable?
	LaplaceIters 	 				= 0
	UseEddy							= false
	dM 								= false //Different terms to use in determining eddy current generating field: dm/dt, dB_ext/dt, dB_demag/dt
	dExt							= false
	dDemag							= false
	B_eddyGeneratingField			*data.Slice //Way too many data slices here
	B_prev							*data.Slice
	m_prev 							*data.Slice
	E_irrotPot 						*data.Slice
	B_eddy 							*data.Slice
	B_eddyF    						= NewVectorField("B_eddy", "T", "Eddy current field", SetEddyField)
	Edens_eddy 						= NewScalarField("Edens_eddy", "J/m3", "Eddy energy density", AddEdens_eddy)
	E_eddy    	 					= NewScalarValue("E_eddy", "J", "Total eddy current field energy", GetEddyEnergy)
	AddEdens_eddy 					= makeEdensAdder(B_eddyF, -1)
)

func init() {
	DeclFunc("InitEddy", InitEddy, "initializes the eddy current environment")
	DeclFunc("SetConductivity", setConductivity, "set material conductivity (uniform)")
	DeclFunc("SetLaplaceIters", SetLaplaceIters, "set the number of iterations for the laplace solver for electric scalar potential")
}

func setConductivity(s float64) {
	Conductivity = s
}

func SetLaplaceIters(s int) {
	LaplaceIters = s
}

func InitEddy(m, ext, demag bool) {
	UseEddy = true
	dM = m
	dExt = ext
	dDemag = demag
	size := globalmesh_.Size()
	B_eddy  = cuda.NewSlice(3, size) //Not sure if this is a good idea, since it takes up some of the GPU memory. 
	B_prev  = cuda.NewSlice(3, size)
	m_prev  = cuda.NewSlice(3, size)
	E_irrotPot = cuda.NewSlice(1, size)
	B_eddyGeneratingField = cuda.NewSlice(3, size)
	cumulativeMagChange = cuda.NewSlice(1, globalmesh_.Size())
	interpTotalDemagField = cuda.NewSlice(3, globalmesh_.Size())

	for c := 0; c < 3; c++ {
		cuda.Zero1_async(B_eddyGeneratingField.Comp(c))
		cuda.Zero1_async(B_eddy.Comp(c))
		cuda.Zero1_async(B_prev.Comp(c))
		cuda.Zero1_async(m_prev.Comp(c))
	}
	cuda.Zero1_async(E_irrotPot)
	registerEnergy(GetEddyEnergy, AddEdens_eddy)
	SetExtendedSolver(8) //Default to extended RK45DP
	StoreFieldsAndMagForEddy(M.Buffer(), sliderLocation)
}

func StoreFieldsAndMagForEddy(m *data.Slice, moved data.Vector) {
	for c := 0; c < 3; c++ {
		cuda.Zero1_async(B_prev.Comp(c))
	}
	if(dM == true){
		data.Copy(m_prev, m)
	}
	if(dDemag == true) {
		SetDemagFieldWithMovement(B_prev, moved) //this has to be done first since it sets the field, doesn't add to it.
	} 
	if(dExt == true) {
		B_ext.AddTo(B_prev)
	}
}

func DetermineEddyGeneratingField(dst, m *data.Slice, dt float64) {
	B_diff := cuda.Buffer(3, globalmesh_.Size())
	defer cuda.Recycle(B_diff)
	m_diff := cuda.Buffer(3, globalmesh_.Size())
	defer cuda.Recycle(m_diff)
	for c := 0; c < 3; c++ {
		cuda.Zero1_async(B_eddyGeneratingField.Comp(c))
		cuda.Zero1_async(B_diff.Comp(c))
		cuda.Zero1_async(m_diff.Comp(c))
	}
	if(dM) {
		cuda.Madd2(m_diff, m, m_prev, float32(1.0), float32(-1.0))
		for c := 0; c < 3; c++ {
			cuda.MSliceMul(m_diff.Comp(c), m_diff.Comp(c), Msat.MSlice()) //scale by msat
		}
	} 
	if(dDemag) {
		data.Copy(B_eddyGeneratingField, dst) //Only demag field calculated by this point of simulation step, so we can just copy
	} 
	if(dExt) {
		B_ext.AddTo(B_eddyGeneratingField)
	}
	if(dDemag || dExt) {
		cuda.Madd2(B_diff, B_eddyGeneratingField, B_prev, float32(1.0), float32(-1.0))
	}
	cuda.Madd2(B_eddyGeneratingField, B_diff, m_diff, float32(-1.0/dt), float32(-1.0*mag.Mu0/dt)) //Approximated time derivatives, m_diff changed into Teslas here. 
}

func CalcEddyField(dst, m, geometry *data.Slice, moving int, dt float64, moved data.Vector) {	
	for c := 0; c < 3; c++ {
		cuda.Zero1_async(B_eddy.Comp(c))
	}
	DetermineEddyGeneratingField(dst, m, dt)
	check := cuda.MaxVecNorm(B_eddyGeneratingField)
	if(check != 0) {					//if there's no eddy generating field, no need to try to solve anything
		SolveFields(dst, geometry, moving)
	}
}

//Solve the irrotational and solenoidal electric field separately, and combine to get eddy currents and the magnetic field generated by them.
//The approach taken from  L. Torres et al, Physica B: Condensed Matter 343, 257 (2004), proceedings of the Fourth Intional Conference on Hysteresis and Micromagnetic Modeling
func SolveFields(dst, geometry *data.Slice, moving int) {
	cuda.Zero1_async(E_irrotPot)
	size := globalmesh_.Size()
	E_Solenoidal := cuda.Buffer(3, size)
	E_Irrot := cuda.Buffer(3, size)
	defer cuda.Recycle(E_Solenoidal)
	defer cuda.Recycle(E_Irrot)
	for c := 0; c < 3; c++ {
		cuda.Zero1_async(E_Solenoidal.Comp(c))
		cuda.Zero1_async(E_Irrot.Comp(c))
	}
	msat := Msat.MSlice()
	defer msat.Recycle()
	spConv().Exec3DSolenoidal(E_Solenoidal, B_eddyGeneratingField, geometry, false, 2, msat) //Solenoidal field, generated by the change in local fields everywhere
	SolveLaplace(E_Solenoidal, E_Irrot, geometry, moving) //Irrotational field with solenoidal used as the boundary condition
	cuda.Madd2(E_Irrot, E_Irrot, E_Solenoidal, float32(Conductivity), float32(Conductivity)) //Total electric field * conductivity = J (reuse E_Irrot here as J)
	spConv().Exec3DSolenoidal(B_eddy, E_Irrot, geometry, false, 0, msat) //New multiplication to get B_eddy from J (J only inside geometry!)
	cuda.Madd2(B_eddy, B_eddy, E_Irrot, mag.Mu0, 0.0) //E_Irrot used as a dummy field here, we only want to multiply B_eddy with Mu_0. TODO: Is there a simpler way for this?
}

func AddEddyField(dst *data.Slice) {
	if(UseEddy) {
		cuda.Madd2(dst, dst, B_eddy, 1.0, 1.0)
	}
}

func SetEddyField(dst *data.Slice) {
	if(UseEddy) {
		data.Copy(dst, B_eddy)
	}
}

//Solve the electric scalar potential Laplacian through successive-over-relaxation on the GPU, using the irrotational field as a Neumann boundary condition.
//TODO: see if there's a harder, better, faster, stronger method for doing this. Also, some sort of convergence checking instead of flat number of iterations.
func SolveLaplace(E_Solenoidal, E_Irrot, geometry *data.Slice, moving int) {
	size := globalmesh_.Size()
	for i := 0; i < LaplaceIters; i++ {
			cuda.Ext_EddyLaplace_async(E_irrotPot, E_Solenoidal, globalmesh_.PBC(), size, 1, geometry, moving, float32(globalmesh_.CellSize()[X]))
			cuda.Ext_EddyLaplace_async(E_irrotPot, E_Solenoidal, globalmesh_.PBC(), size, 0, geometry, moving, float32(globalmesh_.CellSize()[X]))
	}
	cuda.Ext_EddyGradient_async(E_irrotPot, E_Irrot, E_Solenoidal, globalmesh_.PBC(), size, geometry, moving, float32(globalmesh_.CellSize()[X]))
}

func GetEddyEnergy() float64 {
	return -1 * cellVolume() * dot(&M_full, &B_eddyF)
}
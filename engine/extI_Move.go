package engine

import (
	"fmt"
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"time"
	"math"
)


////////////////////////////////////////////////////////////////////////// MoveMain /////////////////////////////////////////////////////////////////////////////////
//Movement-related functions and measurables. There is some code duplication in order to make this orthogonal to the original Mumax3 code
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Movement globals
var (
	movementScheme					= -1 			 //0 - Discrete movement, 1 - Componentwise interpolated field, 2 - Scalar potential method + Interpolation
	moving						= false
	sliderLocation  				= Vector(0.0,0.0,0.0) //The location of the slider. When wrapping through a periodic boundary, modulo is taken from this.
	sliderSpeed 					= Vector(0.0,0.0,0.0)
	sliderAccl 				 	= Vector(0.0,0.0,0.0)
	sliderMass					= 0.0
	draggerLocation 				= Vector(0.0,0.0,0.0) //the location of the dragger ("dragger" refers to spring pulling the slider forward)
	draggerSpeed					= Vector(0.0,0.0,0.0) 
	draggerSpringConstants				= Vector(0.0,0.0,0.0)
	MovingDirections				= Vector(0,0,0) //It's possible to let slider move in only selected directions (1 if moving allowed, 0 if not)
	velocityDragConstant				= 0.0           //Damping force due to spring = -velocityDragConstant*mass*velocity
	springSystem					= false		 //Whether a spring (true) or constant velocity (false) driving is used.
	printDebugs					= false		 //Print some additional debug information
	invertedMovement				= false	     //NOT IMPLEMENTED YET (at least not properly, never tested)
	interpBaseDemagField				*data.Slice 
	interpTotalDemagField				*data.Slice 			 
	cumulativeMagChange				*data.Slice //used when calculating magnetization change in slider and base
	magDifferenceBase				= NewScalarValue("MD_Base", "%", "Base magnetization change", SaveBaseMDiff)
	magDifferenceSlider			  	= NewScalarValue("MD_Slider", "%", "Slider magnetization change", SaveSliderMDiff)
	dissipatedPower 				= NewScalarValue("P_d", "W", "Dissipated power due to magnetic moment relaxation", SaveDissipatedPower)
	DraggerLocationValue 				= NewVectorValue("D_l", "m", "Dragger location", SaveDraggerLocation)
	SliderLocationValue 				= NewVectorValue("S_l", "m", "Slider location", SaveSliderLocation)
	SliderSpeedValue 				= NewVectorValue("S_s", "m/s", "Slider speed", SaveSliderSpeed)
	springForce 					= NewVectorValue("F_spring", "N", "Spring force", SaveSpringForce)
	magneticForce 					= NewVectorValue("F_m", "N", "Magnetic force the base exerts on the slider", SaveMagneticForce)
	B_ib 						= NewVectorField("B_ib", "T", "base interpolated magnetostatic field", SaveBaseField) //used primarily for interpolation debugging
	B_shifted     					= NewVectorField("B_shifted", "T", "interpolated total magnetostatic field", getInterpolatedField)
	TotalDivergence 				= NewScalarValue("D", "-", "Divergence of magnetic field", CalculateDivergences)
	TotalCurl 					= NewScalarValue("C", "-", "Curl of magnetic field", CalculateCurls)
)

func init() {
	DeclFunc("InitMove", initMove, "initializes the movement environment, also sets movement solver to RK45DP_MOVE")
	DeclFunc("SetExtSolver", SetExtendedSolver, "Sets the solver used for extensions (for movement and eddies only Euler and RK45DP currently available)")
	DeclFunc("StartMove", StartMove, "Starts moving the slider")
	DeclFunc("StopMove", StopMove, "Stops moving the slider")
	DeclFunc("SetSpeed", SetSpeed, "Sets the slider movement speed (m/s)")
	DeclFunc("StartDragging", StartDragging, "Starts dragging with a spring (m/s)")
	DeclFunc("SetSpringConstants", SetSpringConstants, "Set the spring constant of the dragging spring (N/m)")
	DeclFunc("SetMass", SetMass, "Set the mass of the slider (kg)")
	DeclFunc("SetVelocityDragConstant", SetVelocityDragConstant, "Set the drag constant for the velocity (Ns/m)")
	DeclFunc("StopDragging", StopDragging, "stops dragging (sets speed of spring to 0)")
	DeclFunc("PrintDebugInfo", PrintDebugInfo, "Print additional debug info each step")
	DeclFunc("SetMovementScheme", SetMovementScheme, "Sets movement to off (-1), discrete (0), interpolated (1) or scalar potential method (2)")
	DeclFunc("Ext_Run", Ext_Run, "Run the simulation for a time in seconds with the extended solver")
	DeclFunc("DefineMovingGeom", DefineMovingGeom, "Sets the geometry for magnets in relative motion (0 for base, 1 for slider)")
	DeclFunc("Invert movement", InvertMovement, "Makes it so that the base moves instead of the slider") //TODO: implement (idea: Conveyor belt type of movement)
	baseGeom.baseinit()
	sliderGeom.sliderinit()
}

func initMove(scheme int) {
	interpBaseDemagField = data.NewSlice(3, globalmesh_.Size())
	interpTotalDemagField = data.NewSlice(3, globalmesh_.Size())
	B_arbitrary = data.NewSlice(3, globalmesh_.Size())
	cumulativeMagChange = cuda.NewSlice(1, globalmesh_.Size())
	registerEnergy(Ext_move_GetDemagEnergy, AddEdens_demag)
	registerEnergy(Ext_move_removeOldDemagEnergy, AddEdens_demag) //TODO: just remove the original energy term from the list instead of this hack
	SetExtendedSolver(8) //Default to extended RK45DP
	SetMovementScheme(scheme) //0 - Discrete movement, 1 - Componentwise interpolated field, 2 - Scalar potential method + Interpolation
}

const (
	EXT_EULER  = 7
	EXT_DORMANDPRINCE  = 8
)

func SetExtendedSolver(typ int) {
	// free previous solver, if any
	if stepper != nil {
		stepper.Free()
	}
	switch typ {
	default:
		util.Fatalf("SetSolver: unknown solver type: %v", typ)
	case EXT_EULER:
		stepper = new(EULER_EXT)
	case EXT_DORMANDPRINCE:
		stepper = new(RK45DP_EXT)
	solvertype = typ
}
}


// Run the simulation for a number of seconds. Basically copied from original code, just here for orthogonality
func Ext_Run(seconds float64) {
	stop := Time + seconds
	alarm = stop // don't have dt adapt to go over alarm
	move_RunWhile(func() bool { return Time < stop })
}

func InvertMovement() {
	invertedMovement = !invertedMovement
}

// Runs as long as condition returns true, saves output.
func move_RunWhile(condition func() bool) {
	SanityCheck()
	pause = false // may be set by <-Inject
	const output = true
	move_runWhile(condition, output)
	pause = true
}

func move_runWhile(condition func() bool, output bool) {
	DoOutput() // allow t=0 output
	for condition() && !pause {
		select {
		default:
		 	step_move(output)
		// accept tasks form Inject channel
		case f := <-Inject:
			f()
		}
	}
}

func torqueFnExtended(dst, prev *data.Slice, moved data.Vector, dte float64) {
	SetTorqueExtended(dst, prev, moved, dte)
	NEvals++
}

func StartMove(speed data.Vector, x, y, z bool) {
	springSystem = false
	moving = true
	MovingDirections = Vector(0,0,0) //Later on we just multiply speed, acceleration etc. with this vector
	if(x == true) {
		MovingDirections[X] = 1.0
	} 
	if(y == true) {
		MovingDirections[Y] = 1.0
	} 
	if(z == true) {
		MovingDirections[Z] = 1.0
	}
	SetSpeed(speed)
}

func SetMovementScheme(s int) {
	movementScheme = s
}

func StopMove() {
	moving = false
	SetSpeed(Vector(0,0,0))
}

func SetSpeed(s data.Vector) {
	sliderSpeed = s //m/s
}

func SetDraggerSpeed(s data.Vector) {
	draggerSpeed = s //m/s
}

func SetMass(m float64) {
	sliderMass = m //kg
}

func SetSpringConstants(x,y,z float64) {
	draggerSpringConstants[X] = x
	draggerSpringConstants[Y] = y
	draggerSpringConstants[Z] = z
}

func SetVelocityDragConstant(s float64) {
	velocityDragConstant = s
}

func PrintDebugInfo(s bool) {
	printDebugs = s
}

func StartDragging(speed data.Vector, ahead data.Vector, resetPos, x, y, z bool) {
	springSystem = true
	moving = true
	if(resetPos) { //resets the dragger position to the same position as the slider + a given distance ahead
		draggerLocation[X] = sliderLocation[X]+ahead[X]
		draggerLocation[Y] = sliderLocation[Y]+ahead[Y]
		draggerLocation[Z] = sliderLocation[Z]+ahead[Z]
	}
	MovingDirections = Vector(0,0,0)
	if(x == true) {
		MovingDirections[X] = 1.0
	} 
	if(y == true) {
		MovingDirections[Y] = 1.0
	} 
	if(z == true) {
		MovingDirections[Z] = 1.0
	}
	SetDraggerSpeed(speed)
}

func StopDragging() {
	SetDraggerSpeed(Vector(0,0,0))
}

func SaveSpringForce() []float64 {
	sL := sliderLocation
	if(movementScheme == 0){
		sL = EleMul(Floor(EleDiv(sL,globalmesh_.CellSize())),globalmesh_.CellSize())
	}
	forceVector := CalculateSpringForce(draggerLocation, sL)
	var returnArray []float64
	for c := 0; c < 3; c++ {
		returnArray = append(returnArray, forceVector[c])
	}
	return returnArray
}

func step_move(output bool) {
	start := time.Now()
	oldNUndone := NUndone
	stepper.Step()
	success := true
	if oldNUndone != NUndone {
		success = false
	}
	for _, f := range postStep {
		f()
	}
	if output && success {
		DoOutput()
	}
	stop := time.Since(start)
	util.Log(stop)
	if(printDebugs) {
	util.Log(stop)
	fmt.Println("Dragger location: ", draggerLocation)
	fmt.Println("Dragger speed: ", draggerSpeed)
	fmt.Println("Slider location: ", sliderLocation)
	fmt.Println("Distance: ", GetSpringDistance(draggerLocation, sliderLocation))
	fmt.Println("Spring force: ", CalculateSpringForce(draggerLocation, sliderLocation))
	fmt.Println("Slider speed: ", sliderSpeed)
	fmt.Println("Timestep: ", Dt_si)
	fmt.Println("Max demag field strength", cuda.MaxVecNorm(interpTotalDemagField))
	fmt.Println("Max eddy field strength", cuda.MaxVecNorm(B_eddy))
	fmt.Println(LastTorque)
	fmt.Println("average divergence: ", CalculateDivergences())
	fmt.Println("average curl: ", CalculateCurls())
	util.Log(Time)
	}
}


func SetTorqueExtended(dst, prev *data.Slice, moved data.Vector, dte float64) {
	SetLLTorqueExtended(dst, prev, moved, dte)
	AddSTTorque(dst)
	FreezeSpins(dst)
}

func SetLLTorqueExtended(dst, prev *data.Slice, moved data.Vector, dte float64) {
	SetEffectiveFieldExtended(dst, prev, moved, dte)
	alpha := Alpha.MSlice()
	defer alpha.Recycle()
	if Precess {
		cuda.LLTorque(dst, M.Buffer(), dst, alpha) // overwrite dst with torque
	} else {
		cuda.LLNoPrecess(dst, M.Buffer(), dst)
	}
}

func getInterpolatedField(dst *data.Slice) { //field is updated every step, so we just copy the latest. TODO: Find out if this is not a good idea.
	data.Copy(dst, interpTotalDemagField)
}

func updateCumulativeMagnetizationChange(m0, m *data.Slice) {
	temp := cuda.Buffer(3, globalmesh_.Size())
	defer cuda.Recycle(temp)
	cuda.Madd2(temp, m, m0, 1.0, -1.0)
	cuda.Ext_move_CudaLens(temp, temp.Size())
	cuda.Madd2(cumulativeMagChange, cumulativeMagChange, temp.Comp(0), 1.0, 1.0)
}

func GetSpringDistance(dloc, sloc data.Vector) data.Vector {
	return dloc.Sub(sloc)
}

func CalculateTotalForce(dloc, sloc, speed data.Vector) data.Vector {
	magneticForceVector := CalculateMagneticForce()
	dragForce := CalculateVelocityDragForce(speed)
	totalForceVector := CalculateSpringForce(dloc, sloc).Add(magneticForceVector).Add(dragForce)
	return totalForceVector
}

func CalculateSpringForce(dloc, sloc data.Vector) data.Vector {
	distance := GetSpringDistance(dloc, sloc)
	forceVector := EleMul(distance, draggerSpringConstants)
	return forceVector
}

func CalculateVelocityDragForce(speed data.Vector) data.Vector {
	return speed.Mul(-velocityDragConstant).Mul(sliderMass)
}

//TODO: cudafy, this is sloooooow
func CalculateMagneticForce() [3]float64{
	var mVectors [3][][][]float32
	var size [3]int = globalmesh_.Size()
	m_Slice := cuda.Buffer(3, size)
	defer cuda.Recycle(m_Slice)
	SetMFull(m_Slice)
	mVectors = m_Slice.HostCopy().Vectors()
	dipoleStrengthPer2HX := cellVolume()/(2.0*globalmesh_.CellSize()[X]) 
	dipoleStrengthPer2HY := cellVolume()/(2.0*globalmesh_.CellSize()[Y]) 
	dipoleStrengthPer2HZ := cellVolume()/(2.0*globalmesh_.CellSize()[Z])
	B_effSlice := cuda.Buffer(3, size)
	defer cuda.Recycle(B_effSlice)
	totalMagneticForce := [3]float64{0.0,0.0,0.0}
	if(UseEddy) {
		cuda.Madd2(B_effSlice, interpBaseDemagField, B_eddy, 1.0, 1.0)
	} else {
		data.Copy(B_effSlice, interpBaseDemagField)
	}
	sliderGeomi, cycle := sliderGeom.Slice()
	if(cycle) {
		defer cuda.Recycle(sliderGeomi)
	}
	sliderGeomiS := sliderGeomi.HostCopy().Scalars()
	B_effVectorField := B_effSlice.HostCopy().Vectors()
	var BFieldCentralDifferenceX [3]float64 = [3]float64{0.0,0.0,0.0}
	var BFieldCentralDifferenceY [3]float64 = [3]float64{0.0,0.0,0.0}
	var BFieldCentralDifferenceZ [3]float64 = [3]float64{0.0,0.0,0.0}
	for x:= 0; x < size[X]; x++ {
		nx := intMod(x+1,size[X])
		px := intMod(x-1+size[X],size[X])
	for y:= 0; y < size[Y]; y++ {
		ny := intMod(y+1,size[Y])
		py := intMod(y-1+size[Y],size[Y])
	for z:= 0; z < size[Z]; z++ {
		nz := intMod(z+1,size[Z])
		pz := intMod(z-1+size[Z],size[Z])
		if(sliderGeomiS[z][y][x] == 2) {
			for c := 0; c < 3; c++ {
				BFieldCentralDifferenceX[c] = float64((B_effVectorField[c][z][y][nx]-B_effVectorField[c][z][y][px])*mVectors[X][z][y][x])
				BFieldCentralDifferenceY[c] = float64((B_effVectorField[c][z][ny][x]-B_effVectorField[c][z][py][x])*mVectors[Y][z][y][x])
				BFieldCentralDifferenceZ[c] = float64((B_effVectorField[c][nz][y][x]-B_effVectorField[c][pz][y][x])*mVectors[Z][z][y][x])
				totalMagneticForce[c] += BFieldCentralDifferenceX[c]*dipoleStrengthPer2HX+BFieldCentralDifferenceY[c]*dipoleStrengthPer2HY+BFieldCentralDifferenceZ[c]*dipoleStrengthPer2HZ
			}
		}
	}}}
	return totalMagneticForce
}


//TODO: Cudafy this
func SaveSliderMDiff() float64 {
	var size [3]int = globalmesh_.Size()
	cmcHost := cumulativeMagChange.HostCopy()
	cmcHostS := cmcHost.Scalars()
	sliderGeomi, cycle := sliderGeom.Slice()
	if(cycle) {
		defer cuda.Recycle(sliderGeomi)
	}
	sliderGeomiS := sliderGeomi.HostCopy().Scalars()
	total := 0.0
	for x:= 0; x < size[X]; x++ {
		for y:= 0; y < size[Y]; y++ {
			for z:= 0; z < size[Z]; z++ {
				if(sliderGeomiS[z][y][x] == 2) {
					total += float64(cmcHostS[z][y][x])
					cmcHostS[z][y][x] = 0
				} 
			}
		}
	}
	data.Copy(cumulativeMagChange, cmcHost)
	return total/(float64(nSliderCells))
}

//TODO: Cudafy this
func SaveBaseMDiff() float64 {
	var size [3]int = globalmesh_.Size()
	cmcHost := cumulativeMagChange.HostCopy()
	cmcHostS := cmcHost.Scalars()
	baseGeomi, cycle := baseGeom.Slice()
	if(cycle) {
		defer cuda.Recycle(baseGeomi)
	}
	baseGeomiS := baseGeomi.HostCopy().Scalars()
	total := 0.0
	for x:= 0; x < size[X]; x++ {
		for y:= 0; y < size[Y]; y++ {
			for z:= 0; z < size[Z]; z++ {
				if(baseGeomiS[z][y][x] == 2) {
					total += float64(cmcHostS[z][y][x])
					cmcHostS[z][y][x] = 0
				} 
			}
		}
	}
	data.Copy(cumulativeMagChange, cmcHost)
	return total/(float64(nBaseCells))
}

func SaveBaseField(dst *data.Slice) {
	data.Copy(dst,interpBaseDemagField)
}

func SaveDraggerLocation() []float64 {
	return []float64{draggerLocation[X],draggerLocation[Y],draggerLocation[Z]}
}


func SaveSliderLocation() []float64 {
	sL := sliderLocation
	if(movementScheme == 0){ //if discrete movement, location is discrete too.
		sL = EleMul(Floor(EleDiv(sL,globalmesh_.CellSize())), globalmesh_.CellSize())
	}
	return []float64{sL[X], sL[Y], sL[Z]}
}

func SaveSliderSpeed() []float64 {
	return []float64{sliderSpeed[X], sliderSpeed[Y], sliderSpeed[Z]}
}


func CalculateDivergences() float64{
	var size [3]int = globalmesh_.Size()
	divergence := cuda.Buffer(1, size)
	defer cuda.Recycle(divergence)
	cuda.Zero1_async(divergence)

	sliderGeomi, _ := sliderGeom.Slice()

	cuda.Ext_move_DivergenceCalc_async(interpBaseDemagField, sliderGeomi, divergence, size[X], size[Y], size[Z], float32(globalmesh_.CellSize()[X]))
	cuda.Ext_move_CudaAbs(divergence, size)
	total := cuda.Sum(divergence)
	total = total/float32(nSliderCells)
	return float64(total)
}

func CalculateCurls() float64{
	var size [3]int = globalmesh_.Size()
	curl := cuda.Buffer(3, size)
	defer cuda.Recycle(curl)
	for c := 0; c < 3; c++ {
			cuda.Zero1_async(curl.Comp(c))
	}

	sliderGeomi, _ := sliderGeom.Slice()

	cuda.Ext_move_CurlCalc_async(interpBaseDemagField, sliderGeomi, curl, size[X], size[Y], size[Z], float32(globalmesh_.CellSize()[X]))
	cuda.Ext_move_CudaLens(curl, size)
	total := cuda.Sum(curl.Comp(0))/float32(nSliderCells)
	return float64(total)
}

//Dissipated power calculation using the formula from Magiera et al. M., EPL (Europhysics Letters) 87, 26002 (2009). 
func CalculateDissipatedPower() float64 { 
	size :=  globalmesh_.Size()
	msat := Msat.MSlice()
	alpha := Alpha.MSlice()

	powers := cuda.Buffer(1, size)
	defer cuda.Recycle(powers)
	B_total := cuda.Buffer(3, size)
	defer cuda.Recycle(B_total)

	cuda.Zero1_async(powers)

	SetEffectiveFieldExtended(B_total, M.Buffer(), sliderLocation, Dt_si)
	cuda.Ext_move_dissipatedPowers(powers, M.Buffer(), B_total, alpha, msat, cellVolume(), GammaLL, size)
	totalPower := float64(cuda.Sum(powers))
	return totalPower
}

func SaveMagneticForce() []float64{
	magneticForce := CalculateMagneticForce()
	var returnArray []float64
	for c := 0; c < 3; c++ {
		returnArray = append(returnArray, magneticForce[c])
	}
	return returnArray
}

func SaveDissipatedPower() float64{ //Interface function, not sure if I need to have this and CalculateDissipatedPower separately.
	return CalculateDissipatedPower()
}

////////////////////////////////////////////////////////////////////////// Solvers //////////////////////////////////////////////////////////////////////////////////
//The movement solvers, in which RK45 solver is used to simultaneously solve the magnetization dynamics and to move slider according to newton's equations of motion.
//Mostly copied from original Mumax RK45 solver, with the movement-related functions added. Same with Euler solver.

//The solution here is naive in the sense that the equations are just both solved in the intermediate steps, and the demag field is interpolated so that the local
//field during the steps is changed according to movement. In the final step all the intermediate steps are combined for both magnetics and movement. Not entirely
//sure if this yields the "correct" solution, or if something more should be taken into account.
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

type RK45DP_EXT struct {
	k1 *data.Slice // torque at end of step is kept for beginning of next step
}

func (rk *RK45DP_EXT) Step() {
	m := M.Buffer()
	size := m.Size()
	if FixDt != 0 {
		Dt_si = FixDt
	}

	// upon resize: remove wrongly sized k1
	if rk.k1.Size() != m.Size() {
		rk.Free()
	}
	// first step ever: one-time k1 init and eval
	if rk.k1 == nil {
		rk.k1 = cuda.NewSlice(3, size)
		torqueFnExtended(rk.k1, m, Vector(0,0,0), Dt_si) //Do not simulate eddy currents here
	}

	// FSAL cannot be used with finite temperature
	if !Temp.isZero() {
		torqueFnExtended(rk.k1, m, Vector(0,0,0), Dt_si)
	}

	t0 := Time
	// backup magnetization
	m0 := cuda.Buffer(3, size)
	defer cuda.Recycle(m0)
	data.Copy(m0, m)

	k2, k3, k4, k5, k6 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)
	defer cuda.Recycle(k2)
	defer cuda.Recycle(k3)
	defer cuda.Recycle(k4)
	defer cuda.Recycle(k5)
	defer cuda.Recycle(k6)
	// k2 will be re-used as k7

	h := float32(Dt_si * GammaLL) // internal time step = Dt * gammaLL
	// there is no explicit stage 1: k1 from previous step
	draggerTempLocation := draggerLocation
	var sliderSpeeds [8]data.Vector
	var sliderAccls [8]data.Vector
	sliderTempLocation := sliderLocation
	sliderSpeeds[1] = sliderSpeed //FSAL
	sliderAccls[1] = sliderAccl
	// stage 2
	Time = t0 + (1./5.)*Dt_si
	cuda.Madd2(m, m, rk.k1, 1, (1./5.)*h) // m = m*1 + k1*h/5
	M.normalize()
	if(moving) {
		if(springSystem) {
			draggerTempLocation = draggerLocation.Add(draggerSpeed.Mul((1./5.)*Dt_si))
			sliderTempLocation = sliderLocation.Add(sliderSpeeds[1].Mul((1./5.)*Dt_si))
			sliderSpeeds[2] = (sliderSpeeds[1].Add(sliderAccls[1].Mul((1./5.)*Dt_si)))
			sliderAccls[2] = EleMul(CalculateTotalForce(draggerTempLocation, sliderTempLocation, sliderSpeeds[2]).Div(sliderMass), MovingDirections)
			
		} else {
			sliderTempLocation = EleMul((sliderLocation.Add(sliderSpeed.Mul((1./5.)*Dt_si))), MovingDirections)
		}
	}
	torqueFnExtended(k2, m, sliderTempLocation, (1./5.)*Dt_si)

	// stage 3
	Time = t0 + (3./10.)*Dt_si
	cuda.Madd3(m, m0, rk.k1, k2, 1, (3./40.)*h, (9./40.)*h)
	M.normalize()
	if(moving) {
		if(springSystem) {
			draggerTempLocation = draggerLocation.Add(draggerSpeed.Mul((3./10.)*Dt_si))
			sliderTempLocation = sliderLocation.Add(sliderSpeeds[1].Mul((3./40.)*Dt_si)).Add(sliderSpeeds[2].Mul((9./40.)*Dt_si))
			sliderSpeeds[3] = (sliderSpeeds[1].Add(sliderAccls[1].Mul((3./40.)*Dt_si)).Add(sliderAccls[2].Mul((9./40.)*Dt_si)))
			sliderAccls[3] = EleMul(CalculateTotalForce(draggerTempLocation, sliderTempLocation, sliderSpeeds[3]).Div(sliderMass), MovingDirections)
		} else {
			sliderTempLocation = EleMul((sliderLocation.Add(sliderSpeed.Mul((3./10.)*Dt_si))), MovingDirections)
		}
	}
	torqueFnExtended(k3, m, sliderTempLocation, (3./10.)*Dt_si)

	// stage 4
	Time = t0 + (4./5.)*Dt_si
	madd4(m, m0, rk.k1, k2, k3, 1, (44./45.)*h, (-56./15.)*h, (32./9.)*h)
	M.normalize()
	if(moving) {
		if(springSystem) {
			draggerTempLocation = draggerLocation.Add(draggerSpeed.Mul((4./5.)*Dt_si))
			sliderTempLocation = sliderLocation.Add(sliderSpeeds[1].Mul((44./45.)*Dt_si)).Add(sliderSpeeds[2].Mul((-56./15.)*Dt_si)).Add(sliderSpeeds[3].Mul((32./9.)*Dt_si))
			sliderSpeeds[4] = (sliderSpeeds[1].Add(sliderAccls[1].Mul((44./45.)*Dt_si)).Add(sliderAccls[2].Mul((-56./15.)*Dt_si)).Add(sliderAccls[3].Mul((32./9.)*Dt_si)))
			sliderAccls[4] = EleMul(CalculateTotalForce(draggerTempLocation, sliderTempLocation, sliderSpeeds[4]).Div(sliderMass), MovingDirections)
		} else {
			sliderTempLocation = EleMul((sliderLocation.Add(sliderSpeed.Mul((4./5.)*Dt_si))), MovingDirections)
		}
	}
	torqueFnExtended(k4, m, sliderTempLocation, (4./5.)*Dt_si)

	// stage 5
	Time = t0 + (8./9.)*Dt_si
	madd5(m, m0, rk.k1, k2, k3, k4, 1, (19372./6561.)*h, (-25360./2187.)*h, (64448./6561.)*h, (-212./729.)*h)
	M.normalize()
	if(moving) {
		if(springSystem) {
			draggerTempLocation = draggerLocation.Add(draggerSpeed.Mul((8./9.)*Dt_si))
			sliderTempLocation = sliderLocation.Add(sliderSpeeds[1].Mul((19372./6561.)*Dt_si)).Add(sliderSpeeds[2].Mul((-25360./2187.)*Dt_si)).Add(sliderSpeeds[3].Mul((64448./6561.)*Dt_si)).Add(sliderSpeeds[4].Mul((-212./729.)*Dt_si))
			sliderSpeeds[5] = (sliderSpeeds[1].Add(sliderAccls[1].Mul((19372./6561.)*Dt_si)).Add(sliderAccls[2].Mul((-25360./2187.)*Dt_si)).Add(sliderAccls[3].Mul((64448./6561.)*Dt_si)).Add(sliderAccls[4].Mul((-212./729.)*Dt_si)))
			sliderAccls[5] = EleMul(CalculateTotalForce(draggerTempLocation, sliderTempLocation, sliderSpeeds[5]).Div(sliderMass), MovingDirections)
		} else {
			sliderTempLocation = EleMul((sliderLocation.Add(sliderSpeed.Mul((8./9.)*Dt_si))), MovingDirections)
		}
	}
	torqueFnExtended(k5, m, sliderTempLocation, (8./9.)*Dt_si)

	// stage 6
	Time = t0 + (1.)*Dt_si
	madd6(m, m0, rk.k1, k2, k3, k4, k5, 1, (9017./3168.)*h, (-355./33.)*h, (46732./5247.)*h, (49./176.)*h, (-5103./18656.)*h)
	M.normalize()
	if(moving) {
		if(springSystem) {
			draggerTempLocation = draggerLocation.Add(draggerSpeed.Mul((1.)*Dt_si))
			sliderTempLocation = sliderLocation.Add(sliderSpeeds[1].Mul((9017./3168.)*Dt_si)).Add(sliderSpeeds[2].Mul((-355./33.)*Dt_si)).Add(sliderSpeeds[3].Mul((46732./5247.)*Dt_si)).Add(sliderSpeeds[4].Mul((49./176.)*Dt_si)).Add(sliderSpeeds[5].Mul((-5103./18656.)*Dt_si))
			sliderSpeeds[6] = (sliderSpeeds[1].Add(sliderAccls[1].Mul((9017./3168.)*Dt_si)).Add(sliderAccls[2].Mul((-355./33.)*Dt_si)).Add(sliderAccls[3].Mul((46732./5247.)*Dt_si)).Add(sliderAccls[4].Mul((49./176.)*Dt_si)).Add(sliderAccls[5].Mul((-5103./18656.)*Dt_si)))
			sliderAccls[6] = EleMul(CalculateTotalForce(draggerTempLocation, sliderTempLocation, sliderSpeeds[6]).Div(sliderMass), MovingDirections)
		} else {
			sliderTempLocation = EleMul((sliderLocation.Add(sliderSpeed.Mul((1.)*Dt_si))), MovingDirections)
		}
	}
	torqueFnExtended(k6, m, sliderTempLocation, (1.)*Dt_si)

	// stage 7: 5th order solution
	Time = t0 + (1.)*Dt_si
	// no k2
	madd6(m, m0, rk.k1, k3, k4, k5, k6, 1, (35./384.)*h, (500./1113.)*h, (125./192.)*h, (-2187./6784.)*h, (11./84.)*h) // 5th
	M.normalize()
	k7 := k2     // re-use k2
	if(moving) {
		if(springSystem) {
			draggerTempLocation = draggerLocation.Add(draggerSpeed.Mul((1.)*Dt_si))
			sliderTempLocation = sliderLocation.Add(sliderSpeeds[1].Mul((35./384.)*Dt_si)).Add(sliderSpeeds[3].Mul((500./1113.)*Dt_si)).Add(sliderSpeeds[4].Mul((125./192.)*Dt_si)).Add(sliderSpeeds[5].Mul((-2187./6784.)*Dt_si)).Add(sliderSpeeds[6].Mul((11./84.)*Dt_si))
			sliderSpeeds[7] = (sliderSpeeds[1].Add(sliderAccls[1].Mul((35./384.)*Dt_si)).Add(sliderAccls[3].Mul((500./1113.)*Dt_si)).Add(sliderAccls[4].Mul((125./192.)*Dt_si)).Add(sliderAccls[5].Mul((-2187./6784.)*Dt_si)).Add(sliderAccls[6].Mul((11./84.)*Dt_si)))
			sliderAccls[7] = EleMul(CalculateTotalForce(draggerTempLocation, sliderTempLocation, sliderSpeeds[7]).Div(sliderMass), MovingDirections)
		} else {
			sliderTempLocation = EleMul((sliderLocation.Add(sliderSpeed.Mul((1.)*Dt_si))), MovingDirections)
		}
	}
	torqueFnExtended(k7, m, sliderTempLocation, (1.)*Dt_si)

	// error estimate
	Err := cuda.Buffer(3, size) //k3 // re-use k3 as error estimate
	defer cuda.Recycle(Err)
	madd6(Err, rk.k1, k3, k4, k5, k6, k7, (35./384.)-(5179./57600.), (500./1113.)-(7571./16695.), (125./192.)-(393./640.), (-2187./6784.)-(-92097./339200.), (11./84.)-(187./2100.), (0.)-(1./40.))

	// determine error
	err := cuda.MaxVecNorm(Err) * float64(h)

	// adjust next time step
	if err < MaxErr || Dt_si <= MinDt || FixDt != 0 { // mindt check to avoid infinite loop
		// step OK
		setLastErr(err)
		setMaxTorque(k7)
		NSteps++
		Time = t0 + Dt_si
		updateCumulativeMagnetizationChange(m0, m)
		if(moving) {
			OldSliderLocationCells := Floor(EleDiv(sliderLocation,globalmesh_.CellSize()))
			sliderLocation = sliderTempLocation
			if(springSystem) {
				draggerLocation = draggerLocation.Add(draggerSpeed.Mul(Dt_si))
				sliderSpeed = sliderSpeeds[7]
				sliderAccl = sliderAccls[7] // FSAL
			}
			NewSliderLocationCells := Floor(EleDiv(sliderTempLocation,globalmesh_.CellSize()))
			wholeCellsmoved := NewSliderLocationCells.Sub(OldSliderLocationCells)
			if(int(wholeCellsmoved[X]) != 0 || int(wholeCellsmoved[Y]) != 0 || int(wholeCellsmoved[Z]) != 0) {
				moveGeometryAndMagnetization(int(wholeCellsmoved[X]), int(wholeCellsmoved[Y]), int(wholeCellsmoved[Z]))
				torqueFnExtended(k7, m, sliderTempLocation, Dt_si) //If we moved a whole cell, recalculate these, likely more accurate
				setMaxTorque(k7)
			}
		}
		if(UseEddy) {
			StoreFieldsAndMagForEddy(m, sliderLocation)
		}
		adaptDt(math.Pow(MaxErr/err, 1./5.))
		data.Copy(rk.k1, k7) // FSAL
	} else {
		// undo bad step
		util.Println("Bad step at t=", t0, ", err=", err)
		util.Assert(FixDt == 0)
		Time = t0
		data.Copy(m, m0)
		NUndone++
		adaptDt(math.Pow(MaxErr/err, 1./6.))
	}
}

func (rk *RK45DP_EXT) Free() {
	rk.k1.Free()
	rk.k1 = nil
}


type EULER_EXT struct{}

// Extended Euler method, can be used as solver.Step.
func (_ *EULER_EXT) Step() {
	m := M.Buffer()
	m0 := cuda.Buffer(3, m.Size())
	defer cuda.Recycle(m0)
	data.Copy(m0, m)
	dm := cuda.Buffer(3, m.Size())
	defer cuda.Recycle(dm)

	torqueFnExtended(dm, m, sliderLocation, Dt_si)
	setMaxTorque(dm)

	// Adaptive time stepping: treat MaxErr as the maximum magnetization delta
	// (proportional to the error, but an overestimation for sure)
	var dt float32
	if FixDt != 0 {
		Dt_si = FixDt
		dt = float32(Dt_si * GammaLL)
	} else {
		dt = float32(MaxErr / LastTorque)
		Dt_si = float64(dt) / GammaLL
	}
	util.AssertMsg(dt > 0, "EULER_EXT solver requires fixed time step > 0")
	setLastErr(float64(dt) * LastTorque)

	if(UseEddy) {
		StoreFieldsAndMagForEddy(m, sliderLocation)
	}

	cuda.Madd2(m, m, dm, 1, dt) // m = m + dt * dm
	M.normalize()

	updateCumulativeMagnetizationChange(m0, m)
	if(moving) {
		OldSliderLocationCells := Floor(EleDiv(sliderLocation,globalmesh_.CellSize()))
		sliderLocation = sliderLocation.Add(sliderSpeed.Mul(Dt_si))
		if(springSystem) {
			draggerLocation = draggerLocation.Add(draggerSpeed.Mul(Dt_si))
			sliderSpeed = (sliderSpeed.Add(sliderAccl.Mul(Dt_si)))
			sliderAccl = EleMul(CalculateTotalForce(draggerLocation, sliderLocation, sliderSpeed).Div(sliderMass), MovingDirections)
		}
		NewSliderLocationCells := Floor(EleDiv(sliderLocation,globalmesh_.CellSize()))
		wholeCellsmoved := NewSliderLocationCells.Sub(OldSliderLocationCells)
		if(int(wholeCellsmoved[X]) != 0 || int(wholeCellsmoved[Y]) != 0 || int(wholeCellsmoved[Z]) != 0) {
			moveGeometryAndMagnetization(int(wholeCellsmoved[X]), int(wholeCellsmoved[Y]), int(wholeCellsmoved[Z]))
		}
	}



	Time += Dt_si
	NSteps++
}

func (_ *EULER_EXT) Free() {}


////////////////////////////////////////////////////////////////////////// Effective Field //////////////////////////////////////////////////////////////////////////
//Effective field calculation with movement, either non-interpolating, direct componentwise interpolation or scalar potential method
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

var B_arbitrary *data.Slice //This is used to observe some *data.Slice during simulation. Purely for debug reasons
var B_arbit = NewScalarField("B_arbit", "T", "Some random field that needs to be studied for debugging", SetArbitField)


func SetArbitField(dst *data.Slice) { 
	data.Copy(dst, B_arbitrary)
}

func SetDemagFieldWithMovement(dst *data.Slice, moved data.Vector) {
	for c := 0; c < 3; c++ {
		cuda.Zero1_async(dst.Comp(c))
	}

	if(movementScheme == -1){ //just normal demag field calculation in the case we're not using any movement functions
		SetDemagField(dst)
		return
	}


	normalizedMoved := EleDiv(EleMod(moved, globalmesh_.CellSize()),globalmesh_.CellSize()) //Movement between cells, normalized to 0..1
	sliderGeomi, cycleSlider := sliderGeom.Slice()
	baseGeomi, cycleBase := baseGeom.Slice()
	if(cycleSlider) {
		defer cuda.Recycle(sliderGeomi)
	}
	if(cycleBase) { 
		defer cuda.Recycle(baseGeomi)
	}
	B_demagSlider := cuda.Buffer(3, globalmesh_.Size())
	defer cuda.Recycle(B_demagSlider)
	for c := 0; c < 3; c++ {
		cuda.Zero1_async(B_demagSlider.Comp(c))
	}
	B_demagBase := cuda.Buffer(3, globalmesh_.Size())
	defer cuda.Recycle(B_demagBase)
	for c := 0; c < 3; c++ {
		cuda.Zero1_async(B_demagBase.Comp(c))
	}

	if(movementScheme == 0){
		if(sliderDefined == true) {
			SetDemagFieldExcludeGen(B_demagBase, sliderGeomi, 0) //take the base field only so it can be used in calculations.
			data.Copy(interpBaseDemagField, B_demagBase)
		}
		SetDemagField(dst)    // set to full B_demag, i.e. both base and slider fields taken into account.
		data.Copy(interpTotalDemagField,dst) //Not actually interpolated in the 0 movementscheme.
		return
	}
	
	if (movementScheme == 1) {
		if(baseDefined == true) {
			//SetDemagFieldExcludeGen(B_demagBase, sliderGeomi, 0) //take the base field only so it can be used in calculations.
			SetDemagFieldExcludeInterp(B_demagBase, normalizedMoved[X], normalizedMoved[Y], normalizedMoved[Z], sliderGeomi, 0)
			data.Copy(interpBaseDemagField, B_demagBase)
		}
		if(sliderDefined == true) {
			//SetDemagFieldExcludeGen(B_demagSlider, sliderGeomi, 1) //take the base field only so it can be used in calculations.
			SetDemagFieldExcludeInterp(B_demagSlider, -normalizedMoved[X], -normalizedMoved[Y], -normalizedMoved[Z], sliderGeomi, 1)
		}
		cuda.Madd2(dst, B_demagSlider, B_demagBase, 1.0, 1.0)
		data.Copy(interpTotalDemagField, dst)
		return
	}

	if (movementScheme == 2) {
		if(baseDefined == true) {
			SetDemagFieldExcludeTotal(B_demagBase, sliderGeomi, 0)
			SetDemagFieldWithScalarPotential(B_demagBase, normalizedMoved[X], normalizedMoved[Y], normalizedMoved[Z], sliderGeomi, 0)
			data.Copy(interpBaseDemagField, B_demagBase)
		}
		if(sliderDefined == true) {
			SetDemagFieldExcludeTotal(B_demagSlider, sliderGeomi, 1)
			SetDemagFieldWithScalarPotential(B_demagSlider, -normalizedMoved[X], -normalizedMoved[Y], -normalizedMoved[Z], sliderGeomi, 1)	
		}
		cuda.Madd2(dst, B_demagSlider, B_demagBase, 1.0, 1.0)
		data.Copy(interpTotalDemagField, dst)
		return
	}
	return
}

// Sets dst to the current effective field (T). The function is extended with possible interpolation due to movement, capability of calculating the demag field via scalar potential and using eddy currents.
func SetEffectiveFieldExtended(dst, m *data.Slice, moved data.Vector, dte float64) {
	
	SetDemagFieldWithMovement(dst, moved)
	if(UseEddy) {
		CalcEddyField(dst, m, geometry.Gpu(), 0, dte, data.Vector{0.0,0.0,0.0}) //No partial movement
		AddEddyField(dst)
	}

	B_ext.AddTo(dst)

	AddExchangeField(dst) // ...then add other terms
	AddAnisotropyField(dst)

	if !relaxing {
		B_therm.AddTo(dst)
	}
}

//Parts of demag.go with changes and additions. Also an older version which still uses Bsat. TODO: update.

var (
	pconv_         		*cuda.PotentialConvolution // does the heavy lifting and provides FFTM
	TestDemag     		= false                // enable convolution self-test
)

func SetDemagFieldWithScalarPotential(dst *data.Slice, mx, my, mz float64, targetGeom *data.Slice, includeOrExclude int) {
	if EnableDemag {
		if NoDemagSpins.isZero() {
			msat := Msat.MSlice()
			defer msat.Recycle()
			// Normal demag, everywhere
			spConv().Exec3D(dst, M.Buffer(), geometry.Gpu(), mx, my, mz, targetGeom, includeOrExclude, globalmesh_.CellSize()[X], msat)
		} else {
			setMaskedPotentialField(dst, mx, my, mz, targetGeom, includeOrExclude)
		}
	} else {
		cuda.Zero(dst) // will ADD other terms to it
	}
}

func SetDemagFieldExcludeGen(dst, targetGeom *data.Slice, includeOrExclude int) {
	if EnableDemag {
		if NoDemagSpins.isZero() {
			// Normal demag, everywhere
			msat := Msat.MSlice()
			defer msat.Recycle()
			cuda.Exec3DExcludeGen(demagConv(), dst, M.Buffer(), geometry.Gpu(), targetGeom, includeOrExclude, msat)
		} else {
			Ext_move_setMaskedDemagFieldExclude(dst, targetGeom, includeOrExclude)
		}
	} else {
		cuda.Zero(dst) // will ADD other terms to it
	}
}

func SetDemagFieldExcludeTotal(dst, targetGeom *data.Slice, includeOrExclude int) {
	if EnableDemag {
		if NoDemagSpins.isZero() {
			msat := Msat.MSlice()
			defer msat.Recycle()
			// Normal demag, everywhere
			cuda.Exec3DExcludeTotal(demagConv(), dst, M.Buffer(), geometry.Gpu(), targetGeom, includeOrExclude, msat)
		} else {
			Ext_move_setMaskedDemagFieldExclude(dst, targetGeom, includeOrExclude)
		}
	} else {
		cuda.Zero(dst) // will ADD other terms to it
	}
}

func SetDemagFieldExcludeInterp(dst *data.Slice, mx, my, mz float64, targetGeom *data.Slice, includeOrExclude int) {
	if EnableDemag {
		if NoDemagSpins.isZero() {
			// Normal demag, everywhere
			msat := Msat.MSlice()
			defer msat.Recycle()
			cuda.Exec3DInterp(demagConv(), dst, M.Buffer(), geometry.Gpu(), mx, my, mz, globalmesh_.CellSize()[X], targetGeom, includeOrExclude, msat)
		} else {
			Ext_move_setMaskedDemagFieldExclude(dst, targetGeom, includeOrExclude)
		}
	} else {
		cuda.Zero(dst) // will ADD other terms to it
	}
}

// Sets dst to the demag field, but cells where NoDemagSpins != 0 do not generate nor recieve field.
func Ext_move_setMaskedDemagFieldExclude(dst, targetGeom *data.Slice, includeOrExclude int) {
	// No-demag spins: mask-out geometry with zeros where NoDemagSpins is set,
	// so these spins do not generate a field

	buf := cuda.Buffer(SCALAR, geometry.Gpu().Size()) // masked-out geometry
	defer cuda.Recycle(buf)

	// obtain a copy of the geometry mask, which we can overwrite
	geom, r := geometry.Slice()
	if r {
		defer cuda.Recycle(geom)
	}
	data.Copy(buf, geom)

	// mask-out
	cuda.ZeroMask(buf, NoDemagSpins.gpuLUT1(), regions.Gpu())
	msat := Msat.MSlice()
	defer msat.Recycle()
	// convolution with masked-out cells.
	cuda.Exec3DExcludeGen(demagConv(), dst, M.Buffer(), buf, dst, 2, msat)

	// After convolution, mask-out the field in the NoDemagSpins cells
	// so they don't feel the field generated by others.
	cuda.ZeroMask(dst, NoDemagSpins.gpuLUT1(), regions.Gpu())
}

func setMaskedPotentialField(dst *data.Slice, mx, my, mz float64, targetGeom *data.Slice, includeOrExclude int) {
	// No-demag spins: mask-out geometry with zeros where NoDemagSpins is set,
	// so these spins do not generate a field

	buf := cuda.Buffer(SCALAR, geometry.Gpu().Size()) // masked-out geometry
	defer cuda.Recycle(buf)

	// obtain a copy of the geometry mask, which we can overwrite
	geom, r := geometry.Slice()
	if r {
		defer cuda.Recycle(geom)
	}
	data.Copy(buf, geom)

	// mask-out
	cuda.ZeroMask(buf, NoDemagSpins.gpuLUT1(), regions.Gpu())
	msat := Msat.MSlice()
	defer msat.Recycle()
	// convolution with masked-out cells.
	spConv().Exec3D(dst, M.Buffer(), buf, mx, my, mz, targetGeom, includeOrExclude, globalmesh_.CellSize()[X], msat)

	// After convolution, mask-out the field in the NoDemagSpins cells
	// so they don't feel the field generated by others.
	cuda.ZeroMask(dst, NoDemagSpins.gpuLUT1(), regions.Gpu())
}

func SetMSat(dst *data.Slice) {
	// scale m by Msat...
	msat, rM := Msat.Slice()
	if rM {
		defer cuda.Recycle(msat)
	}
	for c := 0; c < 3; c++ {
		cuda.Mul(dst.Comp(c), M.Buffer().Comp(c), msat)
	}
}

func spConv() *cuda.PotentialConvolution {
	if pconv_ == nil {
		SetBusy(true)
		defer SetBusy(false)
		kernel := PotentialKernel(Mesh().Size(), Mesh().PBC(), Mesh().CellSize(), DemagAccuracy, *Flag_cachedir)
		pconv_ = cuda.NewPotential(Mesh().Size(), Mesh().PBC(), kernel, TestDemag)
	}
	return pconv_
}

//Hack to remove the old demag energy, since the new one is calculated with the interpolated field
func Ext_move_removeOldDemagEnergy() float64 {
	return 0.5 * cellVolume() * dot(&M_full, &B_demag)
}

// Returns the current demag energy in Joules with the interpolated field.
func Ext_move_GetDemagEnergy() float64 {
	var size [3]int = globalmesh_.Size()
	m_Slice := cuda.Buffer(3, size)
	defer cuda.Recycle(m_Slice)
	SetMFull(m_Slice)

	B := cuda.Buffer(3, size)
	defer cuda.Recycle(B)
	data.Copy(B,interpTotalDemagField)

	return -0.5 * float64(cellVolume()) * float64(cuda.Dot(m_Slice, B))
}

////////////////////////////////////////////////////////////////////////// Moving Geometry //////////////////////////////////////////////////////////////////////////
//Defining moving geometry and the functions related to moving a part of the simulation domain

//Base and Slider are basically just areas that indicate where the respective magnets are, having 2 in their element if the respective magnet has geometry there.
//Additionally, the geometries extend +1 cells in every direction, since we're using finite differences for some calculations over the edge too. In these just outside
//the boundary cells, the value is 1.
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

var (
	baseGeom   						geom
	sliderGeom   					geom
	sliderDefined					= false
	baseDefined 					= false
	sliderOriginCell 		 [3]int = [3]int{0,0,0} //These cells determine the location of the base and slider, from which distances to the dragging spring etc. are calculated.
	baseOriginCell 		 	 [3]int = [3]int{0,0,0} //They are set internally by DefineMovingGeom.
	nSliderCells 				int = 0 //The number of cells in the base and slider, also set internally by DefineMovingGeom
	nBaseCells 				 	int = 0 
)

func zeroGeoms() {
	baseGeom.buffer = cuda.NewSlice(1, geometry.Mesh().Size())
	sliderGeom.buffer = cuda.NewSlice(1, geometry.Mesh().Size())
	cuda.Zero1_async(baseGeom.buffer)
	cuda.Zero1_async(sliderGeom.buffer)
}

func (g *geom) baseinit() {
	g.buffer = nil
	g.info = info{1, "basegeom", ""}
	DeclROnly("basegeom", &baseGeom, "Cell fill fraction (0..1)")
}

func (g *geom) sliderinit() {
	g.buffer = nil
	g.info = info{1, "slidergeom", ""}
	DeclROnly("slidergeom", &sliderGeom, "Cell fill fraction (0..1)")
}

func DefineMovingGeom(s Shape, baseOrSlider int) { //0 for base, 1 for slider
	if(baseDefined == false && sliderDefined == false) { //When defining for the first time, zero both geometries
		zeroGeoms()
	}
	if(baseOrSlider == 0) {
		baseGeom.DefineMovingGeom(s, baseOrSlider)
		baseDefined = true
	} else if (baseOrSlider == 1) {
		sliderGeom.DefineMovingGeom(s, baseOrSlider)
		sliderDefined = true
	}
}

func (geometry *geom) DefineMovingGeom(s Shape, baseOrSlider int) {
	SetBusy(true)
	defer SetBusy(false)
	if s == nil {
		// TODO: would be nice not to save volume if entirely filled
		s = universe
	}

	geometry.shape = s
	if geometry.Gpu().IsNil() {
		geometry.buffer = cuda.NewSlice(1, geometry.Mesh().Size())
	}

	host := data.NewSlice(1, geometry.Gpu().Size())
	array := host.Scalars()
	v := array
	n := geometry.Mesh().Size()

	progress, progmax := 0, n[Y]*n[Z]

	var ok bool
	for iz := 0; iz < n[Z]; iz++ {
		for iy := 0; iy < n[Y]; iy++ {

			progress++
			util.Progress(progress, progmax, "Defining moving geometry")

			for ix := 0; ix < n[X]; ix++ {

				r := Index2Coord(ix, iy, iz)
				x0, y0, z0 := r[X], r[Y], r[Z]

				// check if center and all vertices lie inside or all outside. There's no smoothing with the moving geometry indicator areas
				allIn, allOut := true, true
				if s(x0, y0, z0) {
					allOut = false
				} else {
					allIn = false
				}
				switch {
				case allIn:
					v[iz][iy][ix] = 2
					ok = true
					if(baseOrSlider == 0) {
						baseOriginCell = [3]int{ix,iy,iz}
					} else if(baseOrSlider == 1) {
						sliderOriginCell = [3]int{ix,iy,iz}
					}
				case allOut:
					v[iz][iy][ix] = 0
				default:
					v[iz][iy][ix] = 0
					ok = ok || (v[iz][iy][ix] != 0)
				}
			}
		}
	}


	data.Copy(geometry.buffer, host)
	if(baseOrSlider == 0) {
		nBaseCells = int(cuda.Sum(geometry.buffer))/2
	} else if(baseOrSlider == 1) {
		nSliderCells = int(cuda.Sum(geometry.buffer))/2
	}

	progress, progmax = 0, n[Y]*n[Z]
	for iz := 0; iz < n[Z]; iz++ {
		for iy := 0; iy < n[Y]; iy++ {

			progress++
			util.Progress(progress, progmax, "Extending moving geometry") //Extend the area by 1 cell in every direction. Not the most elegant way of doing this.

			for ix := 0; ix < n[X]; ix++ {

				if(v[iz][iy][ix] == 2) {
					for x := -1; x <= 1; x++ {
						for y := -1; y <= 1; y++ {
							for z := -1; z <= 1; z++ {
								if(v[(iz+z+n[Z])%n[Z]][(iy+y+n[Y])%n[Y]][(ix+x+n[X])%n[X]] == 0) { //wrap around boundaries
									v[(iz+z+n[Z])%n[Z]][(iy+y+n[Y])%n[Y]][(ix+x+n[X])%n[X]] = 1;
								}
							}
						}
					}
				}
			}
		}
	}


	if !ok {
		util.Fatal("SetGeom: geometry completely empty")
	}

	data.Copy(geometry.buffer, host)
}

func (g *geom) shiftZone(dvec [3]int, shiftGeom *data.Slice) {
	s := g.buffer
	s2 := cuda.Buffer(1, g.Mesh().Size())
	defer cuda.Recycle(s2)
	newv := float32(1)
	data.Copy(s2, s)
	cuda.Ext_move_ShiftXZone(s2, s, dvec, newv, newv, shiftGeom)
	data.Copy(s, s2)
}


func ShiftZone(dvec [3]int, shiftGeom *data.Slice) {
	shiftMagZone(M.Buffer(), dvec, shiftGeom)
	regions.shiftZone(dvec, shiftGeom)
	geometry.shiftZone(dvec, shiftGeom)
	M.normalize() //not sure if this is required here
}

func (b *Regions) shiftZone(dvec [3]int, shiftGeom *data.Slice) {
	// TODO: return if no regions defined
	r1 := b.Gpu()
	r2 := cuda.NewBytes(b.Mesh().NCell()) // TODO: somehow recycle
	defer r2.Free()
	newreg := byte(0) // new region at edge
	r2.Copy(r1)
	cuda.Ext_move_ShiftBytesZone(r2, r1, b.Mesh(), dvec, newreg, shiftGeom)
	r1.Copy(r2)
}

func SliceShiftZone(m *data.Slice, dvec [3]int, shiftGeom *data.Slice) {
	m2 := cuda.Buffer(1, m.Size())
	defer cuda.Recycle(m2)
	for c := 0; c < m.NComp(); c++ {
		comp := m.Comp(c)
		data.Copy(m2, comp)
		cuda.Ext_move_ShiftXZone(m2, comp, dvec, float32(ShiftMagL[c]), float32(ShiftMagR[c]), shiftGeom)
		data.Copy(comp, m2) // str0 ?
	}
}

func shiftMagZone(m *data.Slice, dvec [3]int, shiftGeom *data.Slice) {
	m2 := cuda.Buffer(1, m.Size())
	defer cuda.Recycle(m2)
	for c := 0; c < m.NComp(); c++ {
		comp := m.Comp(c)
		data.Copy(m2, comp)
		cuda.Ext_move_ShiftXZone(m2, comp, dvec, float32(ShiftMagL[c]), float32(ShiftMagR[c]), shiftGeom)
		data.Copy(comp, m2) // str0 ?
	}
}

func moveGeometryAndMagnetization(mx, my, mz int) {
	if(invertedMovement) { //inverted movement has not been tested yet.
		baseGeomi, cycle := baseGeom.Slice()
		if(cycle) {
			defer cuda.Recycle(baseGeomi)
		}
		ShiftZone([3]int{-mx, -my, -mz}, baseGeomi)
		baseOriginCell[X] = intMod(baseOriginCell[X]-mx+globalmesh_.Size()[X], globalmesh_.Size()[X]) //Wrap around
		baseOriginCell[Y] = intMod(baseOriginCell[Y]-my+globalmesh_.Size()[Y], globalmesh_.Size()[Y])
		baseOriginCell[Z] = intMod(baseOriginCell[Z]-mz+globalmesh_.Size()[Z], globalmesh_.Size()[Z])
		if(UseEddy) {
			SliceShiftZone(m_prev, [3]int{-mx, -my, -mz}, baseGeomi) //These have to be moved too, since measurables looking at these might get funky otherwise.
			SliceShiftZone(B_prev, [3]int{-mx, -my, -mz}, baseGeomi)
		}
		baseGeom.shiftZone([3]int{-mx, -my, -mz}, baseGeomi)
	} else {
		sliderGeomi, cycle := sliderGeom.Slice()
		if(cycle) {
			defer cuda.Recycle(sliderGeomi)
		}
		ShiftZone([3]int{mx, my, mz}, sliderGeomi)
		sliderOriginCell[X] = intMod(sliderOriginCell[X]+mx+globalmesh_.Size()[X], globalmesh_.Size()[X]) //Wrap around
		sliderOriginCell[Y] = intMod(sliderOriginCell[Y]+my+globalmesh_.Size()[Y], globalmesh_.Size()[Y])
		sliderOriginCell[Z] = intMod(sliderOriginCell[Z]+mz+globalmesh_.Size()[Z], globalmesh_.Size()[Z])
		if(UseEddy) {
			SliceShiftZone(m_prev, [3]int{mx, my, mz}, sliderGeomi)
			SliceShiftZone(B_prev, [3]int{mx, my, mz}, sliderGeomi)
		}
		sliderGeom.shiftZone([3]int{mx, my, mz}, sliderGeomi)
	}
}

////////////////////////////////////////////////////////////////////////// Utility //////////////////////////////////////////////////////////////////////////
//Some utility functions
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

func EleMul(v data.Vector, b data.Vector) data.Vector {
	return data.Vector{b[0] * v[0], b[1] * v[1], b[2] * v[2]}
}

func EleMod(v data.Vector, b data.Vector) data.Vector {
	return data.Vector{math.Mod(v[0], b[0]), math.Mod(v[1], b[1]), math.Mod(v[2], b[2])}
}

func EleDiv(v data.Vector, b data.Vector) data.Vector {
	return data.Vector{v[0]/b[0], v[1]/b[1], v[2]/b[2]}
}

func Floor(v data.Vector) data.Vector {
	return data.Vector{math.Floor(v[0]), math.Floor(v[1]), math.Floor(v[2])}
}

func intMod(modable, modulus int) int{
	if(modable < modulus) {
		return modable
	} else {
		return modable - (modable/modulus)*modulus
	}
}

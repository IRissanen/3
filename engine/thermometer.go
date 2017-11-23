package engine

import "github.com/mumax/3/cuda"

var (
	ThermometerValue = NewScalarValue("Thermometer", "K", "value of thermometer", ReadTemperature)
)

const kB = 1.38064852e-23 // Boltzmann constant

func ReadTemperature() float64 {

	// TODO:
	//   - make it work for non uniform Msat
	//   - measure temperature in different regions seperately
	//   - average out over a time window (fifoRing?)
	//   - check what happens when applying current
	//   - move to stepper in order not to calculate heff twice

	m := M.Buffer()
	size := m.Size()

	// calculate efective field without thermal field
	h := cuda.Buffer(3, size)
	defer cuda.Recycle(h)
	SetDemagField(h)
	AddExchangeField(h)
	AddAnisotropyField(h)
	B_ext.AddTo(h)
	AddCustomField(h)

	mxh := cuda.Buffer(3, size)
	defer cuda.Recycle(mxh)
	cuda.CrossProduct(mxh, m, h)

	div := cuda.Dot(m, h)
	nom := cuda.Dot(mxh, mxh)

	cs := Mesh().CellSize()
	Vcell := cs[X] * cs[Y] * cs[Z]

	return (Vcell * Msat.Average()) / (2 * kB) * float64(nom) / float64(div)
}

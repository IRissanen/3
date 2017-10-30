package engine

import (
	"github.com/mumax/3/data"
	"math"
)

func init() {
	DeclFunc("ext_centerBubble", CenterBubble, "centerBubble shifts m after each step to keep the bubble position close to the center of the window")
}

func centerBubble(x, y, bubblesign int) {
	M := &M
	n := Mesh().Size()

	m := M.Buffer()
	s := m.Size()
	mz := m.Comp(Z).HostCopy().Scalars()[0]
	my := m.Comp(Y).HostCopy().Scalars()[0]
	mx := m.Comp(X).HostCopy().Scalars()[0]

	posx, posy := 0., 0.

	{
		var magsum float32
		var weightedsum float32

		for iy := range mz {
			for ix := range mz[0] {
				if float64(mz[iy][ix]) > 0.5*float64(bubblesign) {
					magsum += ((mz[iy][ix]*float32(-1*float64(bubblesign)) + 1.) / 2.)
					weightedsum += ((mz[iy][ix]*float32(-1*float64(bubblesign)) + 1.) / 2.) * float32(iy)
				}
			}
		}
		posy = float64(weightedsum / magsum)
	}

	{
		var magsum float32
		var weightedsum float32
		magsum = 0.
		weightedsum = 0.

		for ix := range mz[0] {
			for iy := range mz {
				if float64(mz[iy][ix]) > 0.5*float64(bubblesign) {
					magsum += ((mz[iy][ix]*float32(-1*float64(bubblesign)) + 1.) / 2.)
					weightedsum += ((mz[iy][ix]*float32(-1*float64(bubblesign)) + 1.) / 2.) * float32(ix)
				}
			}
		}
		posx = float64(weightedsum / magsum)
	}

	zero := data.Vector{0, 0, 0}
	if ShiftMagL == zero || ShiftMagR == zero || ShiftMagD == zero || ShiftMagU == zero {
		ShiftMagL[Z] = float64(mz[0][0])      //float64(sign)
		ShiftMagR[Z] = float64(mz[0][s[0]-1]) //float64(sign)
		ShiftMagD[Z] = float64(mz[0][0])      //float64(sign)
		ShiftMagU[Z] = float64(mz[s[1]-1][0]) //float64(sign)

		ShiftMagL[Y] = float64(my[0][0])      //float64(sign)
		ShiftMagR[Y] = float64(my[0][s[0]-1]) //float64(sign)
		ShiftMagD[Y] = float64(my[0][0])      //float64(sign)
		ShiftMagU[Y] = float64(my[s[1]-1][0]) //float64(sign)

		ShiftMagL[X] = float64(mx[0][0])      //float64(sign)
		ShiftMagR[X] = float64(mx[0][s[0]-1]) //float64(sign)
		ShiftMagD[X] = float64(mx[0][0])      //float64(sign)
		ShiftMagU[X] = float64(mx[s[1]-1][0]) //float64(sign)

	}
	dx := int(math.Floor(float64(n[X]/2) - posx))
	dy := int(math.Floor(float64(n[Y]/2) - posy))

	//put bubble to center
	if dx != 0 && x == 1 {
		Shift(dx)
	}
	if dy != 0 && y == 1 {
		YShift(dy)
	}

}

// This post-step function centers the simulation window on a bubble
func CenterBubble(x, y, sign int) {
	PostStep(func() { centerBubble(x, y, sign) })
}

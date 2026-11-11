package irisnn

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math"
)

type adamMoments struct{ m, v *mat.Dense }
type adamState struct {
	t       int
	moments map[*mat.Dense]adamMoments
}

func validateOptimizer(name string) error {
	if name != "" && name != "sgd" && name != "adam" {
		return fmt.Errorf("unsupported optimizer %q: use adam or sgd", name)
	}
	return nil
}

// updateParameters advances Adam once for the entire batch, including all biases.
func (nn *NeuralNetwork) updateParameters(parameters, gradients []*mat.Dense) {
	if nn.config.Optimizer != "adam" {
		for i, p := range parameters {
			subtractScaled(p, gradients[i], nn.config.LearningRate)
		}
		return
	}
	if nn.adam == nil {
		nn.adam = &adamState{moments: make(map[*mat.Dense]adamMoments)}
	}
	nn.adam.t++
	correction1 := 1 - math.Pow(.9, float64(nn.adam.t))
	correction2 := 1 - math.Pow(.999, float64(nn.adam.t))
	for i, p := range parameters {
		rows, cols := p.Dims()
		state, ok := nn.adam.moments[p]
		if !ok {
			state = adamMoments{mat.NewDense(rows, cols, nil), mat.NewDense(rows, cols, nil)}
			nn.adam.moments[p] = state
		}
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				g := gradients[i].At(r, c)
				m := .9*state.m.At(r, c) + .1*g
				v := .999*state.v.At(r, c) + .001*g*g
				state.m.Set(r, c, m)
				state.v.Set(r, c, v)
				p.Set(r, c, p.At(r, c)-nn.config.LearningRate*(m/correction1)/(math.Sqrt(v/correction2)+1e-8))
			}
		}
	}
}

package irisnn

import (
	"gonum.org/v1/gonum/mat"
	"math"
	"testing"
)

func TestTwoHiddenGradientAndLearning(t *testing.T) {
	for _, weight := range []float64{1, 3} {
		x := mat.NewDense(4, 2, []float64{-1, -1, -1, -0.5, 1, 0.5, 1, 1})
		y := mat.NewDense(4, 1, []float64{0, 0, 1, 1})
		nn := NewNeuralNetwork(NeuralNetConfig{InputNeurons: 2, OutputNeurons: 1, HiddenNeurons: 3, HiddenNeurons2: 8, NumEpochs: 1, LearningRate: 0.1, Seed: 42, Loss: "bce", PositiveWeight: weight}, nil)
		loss := func() float64 {
			_, p := nn.forward(x)
			v, err := WeightedBinaryCrossEntropy(p, y, weight)
			if err != nil {
				t.Fatal(err)
			}
			return v
		}
		before := loss()
		params := []*mat.Dense{nn.wHidden, nn.bHidden, nn.wHidden2, nn.bHidden2, nn.wOutput, nn.bOutput}
		type entry struct {
			m           *mat.Dense
			r, c        int
			value, grad float64
		}
		var entries []entry
		for _, m := range params {
			rows, cols := m.Dims()
			for r := 0; r < rows; r++ {
				for c := 0; c < cols; c++ {
					v := m.At(r, c)
					eps := 1e-5
					m.Set(r, c, v+eps)
					plus := loss()
					m.Set(r, c, v-eps)
					minus := loss()
					m.Set(r, c, v)
					entries = append(entries, entry{m, r, c, v, (plus - minus) / (2 * eps)})
				}
			}
		}
		if err := nn.Train(x, y); err != nil {
			t.Fatal(err)
		}
		for _, e := range entries {
			got := (e.value - e.m.At(e.r, e.c)) / 0.1
			if math.Abs(got-e.grad) > 1e-7 {
				t.Fatalf("gradient %g, want %g", got, e.grad)
			}
		}
		for i := 0; i < 100; i++ {
			if err := nn.Train(x, y); err != nil {
				t.Fatal(err)
			}
		}
		if after := loss(); after >= before {
			t.Fatalf("BinaryCrossEntropy did not decrease: %g -> %g", before, after)
		}
	}
}

func TestReLUAndRoundTrip(t *testing.T) {
	z := mat.NewDense(1, 3, []float64{-2, 0, 3})
	if !mat.Equal(relu(z), mat.NewDense(1, 3, []float64{0, 0, 3})) || !mat.Equal(reluDerivative(z), mat.NewDense(1, 3, []float64{0, 0, 1})) {
		t.Fatal("ReLU mismatch")
	}
	nn := NewNeuralNetwork(NeuralNetConfig{InputNeurons: 1, HiddenNeurons: 1, HiddenNeurons2: 1, OutputNeurons: 1}, nil)
	nn.wHidden.Set(0, 0, 2)
	nn.wHidden2.Set(0, 0, 3)
	nn.wOutput.Set(0, 0, 0.5)
	x := mat.NewDense(1, 1, []float64{1})
	p, _ := nn.Predict(x)
	if math.Abs(p.At(0, 0)-sigmoid(3)) > 1e-12 {
		t.Fatal(p)
	}
	artifact := nn.ModelArtifact(EvaluationMetrics{}, nil)
	loaded, _, err := LoadModelArtifact(&artifact)
	if err != nil {
		t.Fatal(err)
	}
	q, _ := loaded.Predict(x)
	if !mat.Equal(p, q) {
		t.Fatal("roundtrip changed predictions")
	}
	artifact.Weights[1] = nil
	if _, _, err := LoadModelArtifact(&artifact); err == nil {
		t.Fatal("accepted malformed weights")
	}
}

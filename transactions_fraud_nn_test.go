//go:build ignore

package main

import (
	"go_neural_network/internal/irisnn"
	"gonum.org/v1/gonum/mat"
	"math"
	"path/filepath"
	"testing"
)

func TestBCEGradientAndLearning(t *testing.T) {
	x := mat.NewDense(4, 2, []float64{-1, -1, -1, -0.5, 1, 0.5, 1, 1})
	y := mat.NewDense(4, 1, []float64{0, 0, 1, 1})
	nn := newFraudNeuralNetwork(NNConfig{InputNeurons: 2, OutputNeurons: 1, HiddenNeurons: 2, LearningRate: 0.1, Seed: 42}, nil)
	loss := func() float64 {
		_, p := nn.forward(x)
		v, err := irisnn.BinaryCrossEntropy(p, y)
		if err != nil {
			t.Fatal(err)
		}
		return v
	}
	before := loss()
	params := []*mat.Dense{nn.WeightsHid, nn.BiasHid, nn.WeightsOut, nn.BiasOut}
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
	{
		hidden, p := nn.forward(x)
		nn.backward(x, hidden, p, y, 0)
	}
	for _, e := range entries {
		got := (e.value - e.m.At(e.r, e.c)) / 0.1
		if math.Abs(got-e.grad) > 1e-7 {
			t.Fatalf("gradient %g, want %g", got, e.grad)
		}
	}
	for i := 0; i < 100; i++ {
		{
			hidden, p := nn.forward(x)
			nn.backward(x, hidden, p, y, 0)
		}
	}
	if after := loss(); after >= before {
		t.Fatalf("BCE did not decrease: %g -> %g", before, after)
	}
}

func TestNNModelCompatibility(t *testing.T) {
	for _, width := range []int{1, 2} {
		scaler := &FeatureScaler{Mean: make([]float64, 8), Std: []float64{1, 1, 1, 1, 1, 1, 1, 1}}
		nn := newFraudNeuralNetwork(NNConfig{InputNeurons: 14, OutputNeurons: width, HiddenNeurons: 2, Seed: 42}, scaler)
		tx := FraudTransaction{Type: "PAYMENT", Amount: 10}
		before, label, err := predictTransactionNN(nn, tx)
		if err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(t.TempDir(), "model.json")
		if err := saveFraudModelNN(path, nn); err != nil {
			t.Fatal(err)
		}
		loaded, err := loadFraudModelNN(path)
		if err != nil {
			t.Fatal(err)
		}
		after, afterLabel, err := predictTransactionNN(loaded, tx)
		if err != nil || before != after || label != afterLabel {
			t.Fatalf("width %d roundtrip: %g %g %v", width, before, after, err)
		}
	}
}

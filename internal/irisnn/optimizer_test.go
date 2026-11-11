package irisnn

import (
	"gonum.org/v1/gonum/mat"
	"math"
	"testing"
)

func TestAdamReferenceSteps(t *testing.T) {
	nn := NewNeuralNetwork(NeuralNetConfig{InputNeurons: 1, HiddenNeurons: 1, OutputNeurons: 1, Optimizer: "adam", LearningRate: .001}, nil)
	p := mat.NewDense(1, 1, []float64{1})
	bias := mat.DenseCopyOf(p)
	want, m, v := 1.0, 0.0, 0.0
	for step, g := range []float64{2, -1, 0, 3} {
		m = .9*m + .1*g
		v = .999*v + .001*g*g
		want -= .001 * (m / (1 - math.Pow(.9, float64(step+1)))) / (math.Sqrt(v/(1-math.Pow(.999, float64(step+1)))) + 1e-8)
		gradient := mat.NewDense(1, 1, []float64{g})
		nn.updateParameters([]*mat.Dense{p, bias}, []*mat.Dense{gradient, gradient})
		if nn.adam.t != step+1 || math.Abs(p.At(0, 0)-want) > 1e-14 || !mat.Equal(p, bias) {
			t.Fatal("Adam reference mismatch")
		}
	}
}

func TestAdamBatchLifecycle(t *testing.T) {
	cfg := NeuralNetConfig{InputNeurons: 1, HiddenNeurons: 2, HiddenNeurons2: 2, OutputNeurons: 1, Optimizer: "adam", LearningRate: .001, Loss: "bce", BatchSize: 2, NumEpochs: 2, Seed: 42}
	nn := NewNeuralNetwork(cfg, nil)
	x := mat.NewDense(3, 1, []float64{1, 2, 3})
	y := mat.NewDense(3, 1, []float64{0, 1, 0})
	for call := 1; call <= 2; call++ {
		if err := nn.Train(x, y); err != nil {
			t.Fatal(err)
		}
		if nn.adam.t != 4*call || len(nn.adam.moments) != 6 {
			t.Fatal("incorrect optimizer lifecycle")
		}
	}
	artifact := nn.ModelArtifact(EvaluationMetrics{}, nil)
	loaded, _, err := LoadModelArtifact(&artifact)
	if err != nil || loaded.config.Optimizer != "adam" || loaded.adam != nil {
		t.Fatal("roundtrip", err)
	}
	p, _ := nn.Predict(x)
	q, _ := loaded.Predict(x)
	if !mat.Equal(p, q) {
		t.Fatal("predictions changed")
	}
	artifact.Config.Optimizer = "invalid"
	if _, _, err := LoadModelArtifact(&artifact); err == nil {
		t.Fatal("accepted invalid optimizer")
	}
	nn.config.Optimizer = "invalid"
	if nn.Train(x, y) == nil {
		t.Fatal("accepted invalid optimizer")
	}
	cfg.Optimizer = ""
	a := NewNeuralNetwork(cfg, nil)
	cfg.Optimizer = "sgd"
	b := NewNeuralNetwork(cfg, nil)
	a.Train(x, y)
	b.Train(x, y)
	p, _ = a.Predict(x)
	q, _ = b.Predict(x)
	if !mat.Equal(p, q) {
		t.Fatal("SGD compatibility")
	}
}

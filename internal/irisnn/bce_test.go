package irisnn

import (
	"gonum.org/v1/gonum/mat"
	"math"
	"path/filepath"
	"testing"
)

func TestBinaryCrossEntropyGradientAndLearning(t *testing.T) {
	x := mat.NewDense(4, 2, []float64{-1, -1, -1, -0.5, 1, 0.5, 1, 1})
	y := mat.NewDense(4, 1, []float64{0, 0, 1, 1})
	nn := NewNeuralNetwork(NeuralNetConfig{InputNeurons: 2, OutputNeurons: 1, HiddenNeurons: 2, NumEpochs: 1, LearningRate: 0.1, Seed: 42, Loss: "bce"}, nil)
	loss := func() float64 {
		_, p := nn.forward(x)
		v, err := BinaryCrossEntropy(p, y)
		if err != nil {
			t.Fatal(err)
		}
		return v
	}
	before := loss()
	params := []*mat.Dense{nn.wHidden, nn.bHidden, nn.wOutput, nn.bOutput}
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

func TestBinaryLossAndShapes(t *testing.T) {
	p := mat.NewDense(2, 1, []float64{0.2, 0.8})
	y := mat.NewDense(2, 1, []float64{0, 1})
	loss, err := BinaryCrossEntropy(p, y)
	if err != nil || math.Abs(loss+math.Log(0.8)) > 1e-12 {
		t.Fatalf("BCE %g: %v", loss, err)
	}
	single, err := EvaluateBinaryOutputs(p, y, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	legacy, err := EvaluateBinaryOutputs(mat.NewDense(2, 2, []float64{0.8, 0.2, 0.2, 0.8}), mat.NewDense(2, 2, []float64{1, 0, 0, 1}), 0.5)
	if err != nil || single != legacy {
		t.Fatalf("shape mismatch: %+v %+v %v", single, legacy, err)
	}
	for _, v := range []float64{0, 1, 1e-15, 1 - 1e-15} {
		loss, err := BinaryCrossEntropy(mat.NewDense(2, 1, []float64{v, v}), y)
		if err != nil || math.IsNaN(loss) || math.IsInf(loss, 0) {
			t.Fatalf("invalid extreme BCE: %g %v", loss, err)
		}
	}
	for _, v := range []float64{math.NaN(), math.Inf(1), -0.1, 1.1} {
		if _, err := BinaryPositiveClassProbabilities(mat.NewDense(1, 1, []float64{v})); err == nil {
			t.Fatalf("accepted probability %g", v)
		}
	}
	for _, v := range []float64{-1, 2, 0.5, math.NaN()} {
		if _, err := BinaryActualLabels(mat.NewDense(1, 1, []float64{v})); err == nil {
			t.Fatalf("accepted label %g", v)
		}
	}
	if _, err := BinaryPositiveClassProbabilities(mat.NewDense(1, 3, nil)); err == nil {
		t.Fatal("accepted 3 outputs")
	}
	for _, cfg := range []NeuralNetConfig{
		{InputNeurons: 1, HiddenNeurons: 1, OutputNeurons: 1, Loss: "invalid"},
		{InputNeurons: 1, HiddenNeurons: 1, OutputNeurons: 2, Loss: "bce"},
	} {
		if err := NewNeuralNetwork(cfg, nil).Train(p, y); err == nil {
			t.Fatal("accepted invalid config")
		}
	}
}

func TestBinaryModelRoundTrip(t *testing.T) {
	for _, width := range []int{1, 2} {
		cfg := NeuralNetConfig{InputNeurons: 2, HiddenNeurons: 2, OutputNeurons: width, Seed: 42}
		if width == 1 {
			cfg.Loss = "bce"
			cfg.PositiveWeight = 3
		}
		nn := NewNeuralNetwork(cfg, nil)
		x := mat.NewDense(1, 2, []float64{0.2, 0.7})
		p, err := nn.Predict(x)
		if err != nil {
			t.Fatal(err)
		}
		before, err := BinaryPositiveClassProbabilities(p)
		if err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(t.TempDir(), "model.json")
		if err := nn.SaveModel(path, EvaluationMetrics{}, map[int]string{0: "not_fraud", 1: "fraud"}); err != nil {
			t.Fatal(err)
		}
		loaded, _, artifact, err := LoadModel(path)
		if err != nil {
			t.Fatal(err)
		}
		if artifact.Config.Loss != cfg.Loss || artifact.Config.PositiveWeight != cfg.PositiveWeight {
			t.Fatal("loss configuration changed")
		}
		p, err = loaded.Predict(x)
		if err != nil {
			t.Fatal(err)
		}
		after, err := BinaryPositiveClassProbabilities(p)
		if err != nil || before[0] != after[0] {
			t.Fatalf("width %d roundtrip: %v %v %v", width, before, after, err)
		}
	}
}

func TestWeightedBinaryCrossEntropy(t *testing.T) {
	p := mat.NewDense(2, 1, []float64{0.2, 0.7})
	y := mat.NewDense(2, 1, []float64{0, 1})
	got, err := WeightedBinaryCrossEntropy(p, y, 3)
	want := -(math.Log(0.8) + 3*math.Log(0.7)) / 2
	if err != nil || math.Abs(got-want) > 1e-12 {
		t.Fatalf("loss %g want %g: %v", got, want, err)
	}
	ordinary, _ := BinaryCrossEntropy(p, y)
	one, err := WeightedBinaryCrossEntropy(p, y, 1)
	if err != nil || one != ordinary {
		t.Fatalf("weight 1: %g != %g: %v", one, ordinary, err)
	}
	for _, w := range []float64{0, -1, math.NaN(), math.Inf(1)} {
		if _, err := WeightedBinaryCrossEntropy(p, y, w); err == nil {
			t.Fatalf("accepted weight %g", w)
		}
	}
}

func TestPositiveWeightConfigValidation(t *testing.T) {
	x := mat.NewDense(2, 1, []float64{0, 1})
	y := mat.NewDense(2, 1, []float64{0, 1})
	for _, w := range []float64{-1, math.NaN(), math.Inf(1)} {
		nn := NewNeuralNetwork(NeuralNetConfig{InputNeurons: 1, HiddenNeurons: 2, OutputNeurons: 1, Loss: "bce", PositiveWeight: w}, nil)
		if err := nn.Train(x, y); err == nil {
			t.Fatalf("accepted training weight %g", w)
		}
		artifact := nn.ModelArtifact(EvaluationMetrics{}, nil)
		if _, _, err := LoadModelArtifact(&artifact); err == nil {
			t.Fatalf("accepted saved weight %g", w)
		}
	}
}

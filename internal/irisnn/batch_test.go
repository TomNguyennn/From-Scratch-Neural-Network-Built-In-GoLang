package irisnn

import (
	"errors"
	"gonum.org/v1/gonum/mat"
	"math/rand"
	"testing"
)

func TestBatchesMatchManualUpdates(t *testing.T) {
	x := mat.NewDense(5, 1, []float64{1, 2, 3, 4, 5})
	y := mat.NewDense(5, 1, []float64{0, 1, 0, 1, 0})
	original := mat.DenseCopyOf(x)
	for _, size := range []int{0, 2, 5, 8} {
		cfg := NeuralNetConfig{InputNeurons: 1, HiddenNeurons: 3, HiddenNeurons2: 2, OutputNeurons: 1, Loss: "bce", PositiveWeight: 3, Seed: 42, NumEpochs: 2, LearningRate: .001, BatchSize: size}
		actual := NewNeuralNetwork(cfg, nil)
		expected := NewNeuralNetwork(cfg, nil)
		rng := rand.New(rand.NewSource(42))
		calls := 0
		err := actual.TrainWithEpochCallback(x, y, func(epoch int) error {
			calls++
			if epoch != calls {
				t.Fatal("epoch numbering")
			}
			if size == 0 {
				expected.trainBatch(x, y)
			} else {
				order := rng.Perm(5)
				for start := 0; start < 5; start += size {
					end := min(start+size, 5)
					bx := mat.NewDense(end-start, 1, nil)
					by := mat.NewDense(end-start, 1, nil)
					for row, index := range order[start:end] {
						bx.SetRow(row, x.RawRowView(index))
						by.SetRow(row, y.RawRowView(index))
					}
					expected.trainBatch(bx, by)
				}
			}
			a, _ := actual.Predict(x)
			b, _ := expected.Predict(x)
			if !mat.EqualApprox(a, b, 1e-12) {
				t.Fatalf("batch %d differs", size)
			}
			return nil
		})
		if err != nil || calls != 2 || !mat.Equal(x, original) {
			t.Fatalf("training: %v", err)
		}
	}
	sentinel := errors.New("stop")
	nn := NewNeuralNetwork(NeuralNetConfig{InputNeurons: 1, HiddenNeurons: 2, OutputNeurons: 1, NumEpochs: 2}, nil)
	if err := nn.TrainWithEpochCallback(x, y, func(int) error { return sentinel }); err != sentinel {
		t.Fatal(err)
	}
	nn.config.BatchSize = -1
	if nn.Train(x, y) == nil {
		t.Fatal("accepted negative batch size")
	}
}

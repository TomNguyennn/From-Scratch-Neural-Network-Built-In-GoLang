package tests

import (
	"path/filepath"
	"testing"
	"time"

	"go_neural_network/internal/iris"
	"go_neural_network/internal/irisnn"
	"gonum.org/v1/gonum/mat"
)

func TestIrisModelSaveLoadRoundTrip(t *testing.T) {
	scaler := &irisnn.FeatureScaler{
		Mean: []float64{1, 2, 3, 4},
		Std:  []float64{0.5, 1.5, 2.5, 3.5},
	}
	model := irisnn.NewNeuralNetwork(irisnn.NeuralNetConfig{
		InputNeurons:  4,
		OutputNeurons: 3,
		HiddenNeurons: 6,
		NumEpochs:     10,
		LearningRate:  0.15,
		Seed:          7,
	}, scaler)

	modelPath := filepath.Join(t.TempDir(), "model.json")
	if err := model.SaveModel(modelPath, irisnn.EvaluationMetrics{
		TrainLoss: 0.1,
		TestAcc:   95.0,
		Timestamp: time.Now(),
	}, iris.DefaultLabelNames()); err != nil {
		t.Fatalf("save model: %v", err)
	}

	loadedModel, loadedScaler, artifact, err := irisnn.LoadModel(modelPath)
	if err != nil {
		t.Fatalf("load model: %v", err)
	}

	if artifact.LabelNames[0] != "Setosa" || artifact.LabelNames[1] != "Versicolor" || artifact.LabelNames[2] != "Virginica" {
		t.Fatalf("unexpected label names: %#v", artifact.LabelNames)
	}
	for i, want := range scaler.Mean {
		if loadedScaler.Mean[i] != want {
			t.Fatalf("loaded mean[%d] = %v, want %v", i, loadedScaler.Mean[i], want)
		}
	}
	for i, want := range scaler.Std {
		if loadedScaler.Std[i] != want {
			t.Fatalf("loaded std[%d] = %v, want %v", i, loadedScaler.Std[i], want)
		}
	}

	output, err := loadedModel.Predict(mat.NewDense(1, 4, []float64{0.1, 0.2, 0.3, 0.4}))
	if err != nil {
		t.Fatalf("predict after load: %v", err)
	}
	rows, cols := output.Dims()
	if rows != 1 || cols != 3 {
		t.Fatalf("prediction dims = (%d, %d), want (1, 3)", rows, cols)
	}
}

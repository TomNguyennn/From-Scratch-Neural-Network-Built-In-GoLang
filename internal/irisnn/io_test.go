package irisnn

import (
	"path/filepath"
	"testing"
	"time"

	"gonum.org/v1/gonum/mat"
)

func TestSaveLoadModelRoundTrip(t *testing.T) {
	scaler := &FeatureScaler{
		Mean: []float64{1, 2, 3, 4},
		Std:  []float64{0.5, 1.5, 2.5, 3.5},
	}
	config := NeuralNetConfig{
		InputNeurons:  4,
		OutputNeurons: 3,
		HiddenNeurons: 6,
		NumEpochs:     10,
		LearningRate:  0.15,
		Seed:          7,
	}

	model := NewNeuralNetwork(config, scaler)
	metrics := EvaluationMetrics{
		TrainLoss: 0.1,
		TestAcc:   95.0,
		Timestamp: time.Now(),
	}

	modelPath := filepath.Join(t.TempDir(), "model.json")
	if err := model.SavePreTrainedModel(modelPath, metrics); err != nil {
		t.Fatalf("SavePreTrainedModel returned error: %v", err)
	}

	loadedModel, loadedScaler, artifact, err := LoadModel(modelPath)
	if err != nil {
		t.Fatalf("LoadModel returned error: %v", err)
	}

	if artifact.LabelNames[0] != "Setosa" || artifact.LabelNames[1] != "Versicolor" || artifact.LabelNames[2] != "Virginica" {
		t.Fatalf("unexpected label names: %#v", artifact.LabelNames)
	}

	for i, want := range scaler.Mean {
		if loadedScaler.Mean[i] != want {
			t.Fatalf("loaded scaler mean[%d] = %v, want %v", i, loadedScaler.Mean[i], want)
		}
	}
	for i, want := range scaler.Std {
		if loadedScaler.Std[i] != want {
			t.Fatalf("loaded scaler std[%d] = %v, want %v", i, loadedScaler.Std[i], want)
		}
	}

	input := mat.NewDense(1, 4, []float64{0.1, 0.2, 0.3, 0.4})
	output, err := loadedModel.Predict(input)
	if err != nil {
		t.Fatalf("Predict returned error after LoadModel: %v", err)
	}

	rows, cols := output.Dims()
	if rows != 1 || cols != 3 {
		t.Fatalf("prediction dims = (%d, %d), want (1, 3)", rows, cols)
	}
}

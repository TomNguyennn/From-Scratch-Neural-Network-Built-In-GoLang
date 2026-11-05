package tests

import (
	"path/filepath"
	"testing"
	"time"

	"go_neural_network/internal/iris"
	"go_neural_network/internal/irisnn"
)

func TestPredictBatchPreservesInputOrder(t *testing.T) {
	tempDir := t.TempDir()
	modelPath := filepath.Join(tempDir, "model.json")
	inputPath := filepath.Join(tempDir, "input.csv")
	outputPath := filepath.Join(tempDir, "predictions.csv")

	scaler := &irisnn.FeatureScaler{
		Mean: []float64{0, 0, 0, 0},
		Std:  []float64{1, 1, 1, 1},
	}
	model := irisnn.NewNeuralNetwork(irisnn.NeuralNetConfig{
		InputNeurons:  4,
		OutputNeurons: 3,
		HiddenNeurons: 5,
		NumEpochs:     1,
		LearningRate:  0.1,
		Seed:          42,
	}, scaler)
	if err := model.SaveModel(modelPath, irisnn.EvaluationMetrics{Timestamp: time.Now()}, iris.DefaultLabelNames()); err != nil {
		t.Fatalf("save model: %v", err)
	}

	writeCSVFile(t, inputPath, [][]string{
		{"Id", "SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Setosa", "Versicolor", "Virginica"},
		{"101", "5.1", "3.5", "1.4", "0.2", "1", "0", "0"},
		{"202", "6.3", "2.9", "5.6", "1.8", "0", "0", "1"},
		{"303", "5.7", "2.8", "4.1", "1.3", "0", "1", "0"},
	})

	runGoCommand(
		t,
		"run", "predict.go",
		"--model", modelPath,
		"--batch-input", inputPath,
		"--batch-output", outputPath,
		"--workers", "3",
		"--benchmark-repeats", "1",
	)

	rows := readCSVFile(t, outputPath)
	if len(rows) != 4 {
		t.Fatalf("got %d rows, want 4", len(rows))
	}
	if got := rows[0][8]; got != "PredictedSpecies" {
		t.Fatalf("unexpected prediction header %q", got)
	}
	for i, wantID := range []string{"101", "202", "303"} {
		if got := rows[i+1][0]; got != wantID {
			t.Fatalf("row %d id = %q, want %q", i+1, got, wantID)
		}
		if rows[i+1][8] == "" {
			t.Fatalf("row %d prediction should not be empty", i+1)
		}
	}
}

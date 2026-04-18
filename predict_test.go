package main

import (
	"encoding/csv"
	"os"
	"path/filepath"
	"testing"

	"go_neural_network/internal/irisnn"
)

func newTestPredictor() (*irisnn.NeuralNetwork, *irisnn.FeatureScaler, map[int]string) {
	scaler := &irisnn.FeatureScaler{
		Mean: []float64{0, 0, 0, 0},
		Std:  []float64{1, 1, 1, 1},
	}
	config := irisnn.NeuralNetConfig{
		InputNeurons:  4,
		OutputNeurons: 3,
		HiddenNeurons: 5,
		NumEpochs:     1,
		LearningRate:  0.1,
		Seed:          42,
	}

	labelNames := map[int]string{
		0: "Setosa",
		1: "Versicolor",
		2: "Virginica",
	}

	return irisnn.NewNeuralNetwork(config, scaler), scaler, labelNames
}

func TestExtractFeaturesWithIDRow(t *testing.T) {
	row := []string{"7", "5.1", "3.5", "1.4", "0.2", "1", "0", "0"}

	features, err := extractFeatures(row)
	if err != nil {
		t.Fatalf("extractFeatures returned error: %v", err)
	}

	expected := []float64{5.1, 3.5, 1.4, 0.2}
	for i, want := range expected {
		if features[i] != want {
			t.Fatalf("feature %d = %v, want %v", i, features[i], want)
		}
	}
}

func TestExtractFeaturesWithPlainFeatureRow(t *testing.T) {
	row := []string{"6.4", "3.2", "4.5", "1.5"}

	features, err := extractFeatures(row)
	if err != nil {
		t.Fatalf("extractFeatures returned error: %v", err)
	}

	expected := []float64{6.4, 3.2, 4.5, 1.5}
	for i, want := range expected {
		if features[i] != want {
			t.Fatalf("feature %d = %v, want %v", i, features[i], want)
		}
	}
}

func TestExtractFeaturesRejectsShortRow(t *testing.T) {
	_, err := extractFeatures([]string{"5.1", "3.5", "1.4"})
	if err == nil {
		t.Fatal("expected extractFeatures to reject short row")
	}
}

func TestRunBatchModePreservesInputOrder(t *testing.T) {
	model, scaler, labelNames := newTestPredictor()
	tempDir := t.TempDir()

	inputPath := filepath.Join(tempDir, "input.csv")
	outputPath := filepath.Join(tempDir, "predictions.csv")

	inputFile, err := os.Create(inputPath)
	if err != nil {
		t.Fatalf("create input csv: %v", err)
	}

	writer := csv.NewWriter(inputFile)
	rows := [][]string{
		{"Id", "SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Setosa", "Versicolor", "Virginica"},
		{"101", "5.1", "3.5", "1.4", "0.2", "1", "0", "0"},
		{"202", "6.3", "2.9", "5.6", "1.8", "0", "0", "1"},
		{"303", "5.7", "2.8", "4.1", "1.3", "0", "1", "0"},
	}
	if err := writer.WriteAll(rows); err != nil {
		t.Fatalf("write input csv: %v", err)
	}
	writer.Flush()
	if err := writer.Error(); err != nil {
		t.Fatalf("flush input csv: %v", err)
	}
	if err := inputFile.Close(); err != nil {
		t.Fatalf("close input csv: %v", err)
	}

	if err := runBatchMode(model, scaler, labelNames, inputPath, outputPath, 3, 2); err != nil {
		t.Fatalf("runBatchMode returned error: %v", err)
	}

	outputFile, err := os.Open(outputPath)
	if err != nil {
		t.Fatalf("open output csv: %v", err)
	}
	defer outputFile.Close()

	outputRows, err := csv.NewReader(outputFile).ReadAll()
	if err != nil {
		t.Fatalf("read output csv: %v", err)
	}

	if len(outputRows) != 4 {
		t.Fatalf("got %d output rows, want 4", len(outputRows))
	}

	if got, want := outputRows[0][8], "PredictedSpecies"; got != want {
		t.Fatalf("header column = %q, want %q", got, want)
	}

	for i, wantID := range []string{"101", "202", "303"} {
		if got := outputRows[i+1][0]; got != wantID {
			t.Fatalf("row %d id = %q, want %q", i+1, got, wantID)
		}
		if outputRows[i+1][8] == "" {
			t.Fatalf("row %d prediction should not be empty", i+1)
		}
		if len(outputRows[i+1]) != 12 {
			t.Fatalf("row %d has %d columns, want 12", i+1, len(outputRows[i+1]))
		}
	}
}

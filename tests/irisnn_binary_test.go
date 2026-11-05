package tests

import (
	"math"
	"path/filepath"
	"testing"
	"time"

	"go_neural_network/internal/irisnn"
	"gonum.org/v1/gonum/mat"
)

func TestGenericModelArtifactSupportsNonIrisShapes(t *testing.T) {
	scaler := &irisnn.FeatureScaler{
		Mean: []float64{10, 20},
		Std:  []float64{2, 4},
	}
	model := irisnn.NewNeuralNetwork(irisnn.NeuralNetConfig{
		InputNeurons:  2,
		OutputNeurons: 2,
		HiddenNeurons: 4,
		NumEpochs:     5,
		LearningRate:  0.05,
		Seed:          11,
	}, scaler)

	modelPath := filepath.Join(t.TempDir(), "binary-model.json")
	labelNames := map[int]string{
		0: "legit",
		1: "fraud",
	}
	if err := model.SaveModel(modelPath, irisnn.EvaluationMetrics{Timestamp: time.Now()}, labelNames); err != nil {
		t.Fatalf("save generic model: %v", err)
	}

	loadedModel, loadedScaler, artifact, err := irisnn.LoadModel(modelPath)
	if err != nil {
		t.Fatalf("load generic model: %v", err)
	}

	if artifact.InputWidth() != 2 {
		t.Fatalf("artifact input width = %d, want 2", artifact.InputWidth())
	}
	if artifact.OutputWidth() != 2 {
		t.Fatalf("artifact output width = %d, want 2", artifact.OutputWidth())
	}
	if artifact.LabelNames[1] != "fraud" {
		t.Fatalf("artifact label names = %#v, want fraud class label", artifact.LabelNames)
	}
	if loadedScaler.Mean[1] != 20 || loadedScaler.Std[0] != 2 {
		t.Fatalf("loaded scaler = %#v, want original stats", loadedScaler)
	}

	output, err := loadedModel.Predict(mat.NewDense(1, 2, []float64{0.5, -0.25}))
	if err != nil {
		t.Fatalf("predict after generic load: %v", err)
	}
	rows, cols := output.Dims()
	if rows != 1 || cols != 2 {
		t.Fatalf("prediction dims = (%d, %d), want (1, 2)", rows, cols)
	}
}

func TestBinaryHelpersBuildAndEvaluateDatasets(t *testing.T) {
	inputs := mat.NewDense(3, 2, []float64{
		1, 10,
		2, 20,
		3, 30,
	})
	labels := []int{0, 1, 1}

	dataset, err := irisnn.NewBinaryDataset(inputs, labels)
	if err != nil {
		t.Fatalf("build binary dataset: %v", err)
	}
	if dataset.Rows != 3 {
		t.Fatalf("dataset rows = %d, want 3", dataset.Rows)
	}
	labelRows, labelCols := dataset.Labels.Dims()
	if labelRows != 3 || labelCols != 2 {
		t.Fatalf("label dims = (%d, %d), want (3, 2)", labelRows, labelCols)
	}

	predictions := mat.NewDense(3, 2, []float64{
		0.85, 0.15,
		0.20, 0.80,
		0.55, 0.45,
	})

	probabilities, err := irisnn.BinaryPositiveClassProbabilities(predictions)
	if err != nil {
		t.Fatalf("extract positive class probabilities: %v", err)
	}
	wantProbabilities := []float64{0.15, 0.80, 0.45}
	for i, want := range wantProbabilities {
		if math.Abs(probabilities[i]-want) > 1e-9 {
			t.Fatalf("probability[%d] = %f, want %f", i, probabilities[i], want)
		}
	}

	actuals, err := irisnn.BinaryActualLabels(dataset.Labels)
	if err != nil {
		t.Fatalf("extract binary labels: %v", err)
	}
	if actuals[0] != 0 || actuals[1] != 1 || actuals[2] != 1 {
		t.Fatalf("actuals = %#v, want [0 1 1]", actuals)
	}

	metrics, err := irisnn.EvaluateBinaryOutputs(predictions, dataset.Labels, 0.5)
	if err != nil {
		t.Fatalf("evaluate binary outputs: %v", err)
	}
	if metrics.TruePos != 1 || metrics.TrueNeg != 1 || metrics.FalseNeg != 1 || metrics.FalsePos != 0 {
		t.Fatalf("unexpected confusion matrix: %#v", metrics)
	}
	if math.Abs(metrics.Accuracy-(2.0/3.0)) > 1e-9 {
		t.Fatalf("accuracy = %f, want %f", metrics.Accuracy, 2.0/3.0)
	}
	if math.Abs(metrics.Recall-0.5) > 1e-9 {
		t.Fatalf("recall = %f, want 0.5", metrics.Recall)
	}
}

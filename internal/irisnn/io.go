package irisnn

import (
	"encoding/json"
	"fmt"
	"os"

	"gonum.org/v1/gonum/mat"
)

type ModelArtifact struct {
	Weights     [][]float64       `json:"weights"`
	Biases      [][]float64       `json:"biases"`
	LayerShapes [][]int           `json:"layer_shapes"`
	Config      NeuralNetConfig   `json:"config"`
	ScalerStats *FeatureScaler    `json:"scaler_stats,omitempty"`
	LabelNames  map[int]string    `json:"label_names,omitempty"`
	Metrics     EvaluationMetrics `json:"metrics"`
}

func (artifact *ModelArtifact) InputWidth() int {
	return artifact.Config.InputNeurons
}

func (artifact *ModelArtifact) OutputWidth() int {
	return artifact.Config.OutputNeurons
}

func (nn *NeuralNetwork) ModelArtifact(metrics EvaluationMetrics, labelNames map[int]string) ModelArtifact {
	hwRows, hwCols := nn.wHidden.Dims()
	owRows, owCols := nn.wOutput.Dims()

	return ModelArtifact{
		Weights: [][]float64{
			append([]float64(nil), nn.wHidden.RawMatrix().Data...),
			append([]float64(nil), nn.wOutput.RawMatrix().Data...),
		},
		Biases: [][]float64{
			append([]float64(nil), nn.bHidden.RawMatrix().Data...),
			append([]float64(nil), nn.bOutput.RawMatrix().Data...),
		},
		LayerShapes: [][]int{
			{hwRows, hwCols},
			{owRows, owCols},
		},
		Config:      nn.config,
		ScalerStats: nn.scaler,
		LabelNames:  labelNames,
		Metrics:     metrics,
	}
}

func (nn *NeuralNetwork) SaveModel(filepath string, metrics EvaluationMetrics, labelNames map[int]string) error {
	artifact := nn.ModelArtifact(metrics, labelNames)

	data, err := json.MarshalIndent(artifact, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(filepath, data, 0644)
}

func LoadModel(filepath string) (*NeuralNetwork, *FeatureScaler, *ModelArtifact, error) {
	data, err := os.ReadFile(filepath)
	if err != nil {
		return nil, nil, nil, err
	}

	var artifact ModelArtifact
	if err := json.Unmarshal(data, &artifact); err != nil {
		return nil, nil, nil, err
	}

	nn, scaler, err := LoadModelArtifact(&artifact)
	if err != nil {
		return nil, nil, nil, err
	}

	return nn, scaler, &artifact, nil
}

func LoadModelArtifact(artifact *ModelArtifact) (*NeuralNetwork, *FeatureScaler, error) {
	if artifact == nil {
		return nil, nil, fmt.Errorf("model artifact is nil")
	}

	if len(artifact.LayerShapes) != 2 || len(artifact.Weights) != 2 || len(artifact.Biases) != 2 {
		return nil, nil, fmt.Errorf("invalid model artifact")
	}

	nn := &NeuralNetwork{
		config: artifact.Config,
		scaler: artifact.ScalerStats,
		wHidden: mat.NewDense(
			artifact.LayerShapes[0][0],
			artifact.LayerShapes[0][1],
			artifact.Weights[0],
		),
		bHidden: mat.NewDense(1, len(artifact.Biases[0]), artifact.Biases[0]),
		wOutput: mat.NewDense(
			artifact.LayerShapes[1][0],
			artifact.LayerShapes[1][1],
			artifact.Weights[1],
		),
		bOutput: mat.NewDense(1, len(artifact.Biases[1]), artifact.Biases[1]),
	}

	var scaler *FeatureScaler
	if artifact.ScalerStats != nil {
		scaler = &FeatureScaler{
			Mean: append([]float64(nil), artifact.ScalerStats.Mean...),
			Std:  append([]float64(nil), artifact.ScalerStats.Std...),
		}
		nn.scaler = scaler
	}

	return nn, scaler, nil
}

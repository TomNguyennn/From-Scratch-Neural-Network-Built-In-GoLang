package irisnn

import (
	"encoding/json"
	"fmt"
	"math"
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

	artifact := ModelArtifact{
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
	if nn.config.HiddenNeurons2 > 0 {
		r, c := nn.wHidden2.Dims()
		artifact.Weights = [][]float64{artifact.Weights[0], append([]float64(nil), nn.wHidden2.RawMatrix().Data...), artifact.Weights[1]}
		artifact.Biases = [][]float64{artifact.Biases[0], append([]float64(nil), nn.bHidden2.RawMatrix().Data...), artifact.Biases[1]}
		artifact.LayerShapes = [][]int{artifact.LayerShapes[0], {r, c}, artifact.LayerShapes[1]}
	}
	return artifact
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

	if err := validateOptimizer(artifact.Config.Optimizer); err != nil {
		return nil, nil, err
	}
	if w := artifact.Config.PositiveWeight; w < 0 || math.IsNaN(w) || math.IsInf(w, 0) {
		return nil, nil, fmt.Errorf("invalid positive weight")
	}
	if artifact.Config.BatchSize < 0 {
		return nil, nil, fmt.Errorf("invalid batch size")
	}
	widths := []int{artifact.Config.InputNeurons, artifact.Config.HiddenNeurons}
	if artifact.Config.HiddenNeurons2 < 0 {
		return nil, nil, fmt.Errorf("invalid second hidden width")
	}
	if artifact.Config.HiddenNeurons2 > 0 {
		widths = append(widths, artifact.Config.HiddenNeurons2)
	}
	widths = append(widths, artifact.Config.OutputNeurons)
	n := len(widths) - 1
	if len(artifact.LayerShapes) != n || len(artifact.Weights) != n || len(artifact.Biases) != n {
		return nil, nil, fmt.Errorf("invalid model layer count")
	}
	weights := make([]*mat.Dense, n)
	biases := make([]*mat.Dense, n)
	for i := 0; i < n; i++ {
		shape := artifact.LayerShapes[i]
		if widths[i] <= 0 || widths[i+1] <= 0 || len(shape) != 2 || shape[0] != widths[i] || shape[1] != widths[i+1] || len(artifact.Weights[i])/widths[i] != widths[i+1] || len(artifact.Weights[i])%widths[i] != 0 || len(artifact.Biases[i]) != widths[i+1] {
			return nil, nil, fmt.Errorf("invalid model layer %d", i)
		}
		weights[i] = mat.NewDense(widths[i], widths[i+1], artifact.Weights[i])
		biases[i] = mat.NewDense(1, widths[i+1], artifact.Biases[i])
	}
	nn := &NeuralNetwork{config: artifact.Config, wHidden: weights[0], bHidden: biases[0], wOutput: weights[n-1], bOutput: biases[n-1]}
	if n == 3 {
		nn.wHidden2 = weights[1]
		nn.bHidden2 = biases[1]
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

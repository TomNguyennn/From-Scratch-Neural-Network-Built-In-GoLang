package irisnn

import (
	"encoding/json"
	"os"

	"gonum.org/v1/gonum/mat"
)



type ModelArtifact struct {
	Weights     [][]float64 `json:"weights"`
	Biases      [][]float64 `json:"biases"`
	LayerShapes [][]int     `json:"layer_shapes"`

	Config 	NeuralNetConfig `json:"config"`
	ScalerStats  *FeatureScaler `json:"scaler_stats"`
	LabelNames   map[int]string     `json:"label_names"`
	
	Metrics EvaluationMetrics
}


func (nn *NeuralNetwork) SavePreTrainedModel(filepath string, metrics EvaluationMetrics) error {

	hwRows, hwCols := nn.wHidden.Dims()
	owRows, owCols := nn.wOutput.Dims()

	
	weights := [][]float64{
		nn.wHidden.RawMatrix().Data,
		nn.wOutput.RawMatrix().Data,
	}

	biases := [][]float64{
		nn.bHidden.RawMatrix().Data,
		nn.bOutput.RawMatrix().Data,
	}

	shapes := [][]int{
		{hwRows, hwCols},
		{owRows, owCols},
	}
	
	myLabels := map[int]string{
    0: "Ssetosa",
    1: "Versicolor",
    2: "Virginica",
}

	artifact := ModelArtifact{
		Weights: weights,
		Biases: biases,
		LayerShapes: shapes,
		Config: nn.config,
		ScalerStats: nn.scaler,
		LabelNames: myLabels,
		Metrics: metrics,
	}

	data, err := json.MarshalIndent(artifact, "", "  ")
    if err != nil {
        return err
    }

    return os.WriteFile(filepath, data, 0644)
}

func LoadModel(filepath string) (*NeuralNetwork, error) {
	data, err := os.ReadFile(filepath)

	//io error handling
	if err != nil {
		return nil, err 
	}

	//parsing json stat to new neural net
	var artifact ModelArtifact
	err = json.Unmarshal(data, &artifact)

	if err != nil {
		return nil, err
	}

	hiddenWeights := mat.NewDense(
		artifact.LayerShapes[0][0],
		artifact.LayerShapes[0][1],
		artifact.Weights[0],
	)

	hiddenBiases := mat.NewDense(len(artifact.Biases[0]), 1, artifact.Biases[0])

	outputWeights := mat.NewDense(
		artifact.LayerShapes[1][0], 
		artifact.LayerShapes[1][1], 
		artifact.Weights[1],
	)
	outputBiases := mat.NewDense(len(artifact.Biases[1]), 1, artifact.Biases[1])

	nn := &NeuralNetwork{
		config: artifact.Config,
		scaler: artifact.ScalerStats,
		wHidden: hiddenWeights,
		bHidden: hiddenBiases,
		wOutput: outputWeights,
		bOutput: outputBiases,
	}

	return nn, nil
}

func main(){
	
}



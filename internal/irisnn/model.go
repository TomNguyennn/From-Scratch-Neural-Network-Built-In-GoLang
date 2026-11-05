package irisnn

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

// NeuralNetwork stores the trainable parameters for a simple
// single-hidden-layer classifier.
type NeuralNetwork struct {
	config  NeuralNetConfig
	scaler  *FeatureScaler
	wHidden *mat.Dense
	bHidden *mat.Dense
	wOutput *mat.Dense
	bOutput *mat.Dense
}

type NeuralNetConfig struct {
	Loss          string  `json:"loss,omitempty"`
	InputNeurons  int     `json:"inputNeurons"`
	OutputNeurons int     `json:"outputNeurons"`
	HiddenNeurons int     `json:"hiddenNeurons"`
	NumEpochs     int     `json:"numEpochs"`
	LearningRate  float64 `json:"learningRate"`
	Seed          int     `json:"seed"`
}

type EvaluationMetrics struct {
	TrainLoss float64 `json:"train_loss"`
	ValLoss   float64 `json:"validation_loss"`
	TestLoss  float64 `json:"test_loss"`

	TrainAcc float64 `json:"train_accuracy"`
	ValAcc   float64 `json:"validation_accuracy"`
	TestAcc  float64 `json:"test_accuracy"`

	Duration  time.Duration `json:"training_duration"`
	Timestamp time.Time     `json:"timestamp"`
}

func NewNeuralNetwork(config NeuralNetConfig, scaler *FeatureScaler) *NeuralNetwork {
	rng := rand.New(rand.NewSource(int64(config.Seed)))

	return &NeuralNetwork{
		config:  config,
		scaler:  scaler,
		wHidden: randDense(rng, config.InputNeurons, config.HiddenNeurons),
		bHidden: mat.NewDense(1, config.HiddenNeurons, nil),
		wOutput: randDense(rng, config.HiddenNeurons, config.OutputNeurons),
		bOutput: mat.NewDense(1, config.OutputNeurons, nil),
	}
}

func randDense(rng *rand.Rand, rows, cols int) *mat.Dense {
	data := make([]float64, rows*cols)
	scale := 0.5
	for i := range data {
		data[i] = (rng.Float64()*2 - 1) * scale
	}
	return mat.NewDense(rows, cols, data)
}

func sigmoid(x float64) float64 {
	if x >= 0 {
		z := math.Exp(-x)
		return 1 / (1 + z)
	}
	z := math.Exp(x)
	return z / (1 + z)
}

func sigmoidPrimeFromActivation(a float64) float64 {
	return a * (1.0 - a)
}

func applySigmoid(src *mat.Dense) *mat.Dense {
	rows, cols := src.Dims()
	out := mat.NewDense(rows, cols, nil)
	out.Apply(func(_, _ int, v float64) float64 {
		return sigmoid(v)
	}, src)
	return out
}

func addBias(m, bias *mat.Dense) {
	rows, cols := m.Dims()
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			m.Set(r, c, m.At(r, c)+bias.At(0, c))
		}
	}
}

func subtractScaled(dst, grad *mat.Dense, scale float64) {
	rows, cols := dst.Dims()
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			dst.Set(r, c, dst.At(r, c)-scale*grad.At(r, c))
		}
	}
}

func MeanSquaredError(predictions, labels *mat.Dense) float64 {
	rows, cols := predictions.Dims()
	var sum float64
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			diff := predictions.At(r, c) - labels.At(r, c)
			sum += diff * diff
		}
	}
	return sum / float64(rows*cols)
}

func sumAlongAxis0(m *mat.Dense) *mat.Dense {
	rows, cols := m.Dims()
	data := make([]float64, cols)
	for c := 0; c < cols; c++ {
		var sum float64
		for r := 0; r < rows; r++ {
			sum += m.At(r, c)
		}
		data[c] = sum
	}
	return mat.NewDense(1, cols, data)
}

func scaleDense(m *mat.Dense, factor float64) {
	rows, cols := m.Dims()
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			m.Set(r, c, m.At(r, c)*factor)
		}
	}
}

func elementwiseSigmoidPrime(m *mat.Dense) *mat.Dense {
	rows, cols := m.Dims()
	out := mat.NewDense(rows, cols, nil)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			out.Set(r, c, sigmoidPrimeFromActivation(m.At(r, c)))
		}
	}
	return out
}

func elementwiseMultiply(a, b *mat.Dense) *mat.Dense {
	rows, cols := a.Dims()
	out := mat.NewDense(rows, cols, nil)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			out.Set(r, c, a.At(r, c)*b.At(r, c))
		}
	}
	return out
}

func subtractDense(a, b *mat.Dense) *mat.Dense {
	rows, cols := a.Dims()
	out := mat.NewDense(rows, cols, nil)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			out.Set(r, c, a.At(r, c)-b.At(r, c))
		}
	}
	return out
}

func argmaxRow(m *mat.Dense, row int) int {
	_, cols := m.Dims()
	bestIndex := 0
	bestValue := m.At(row, 0)
	for c := 1; c < cols; c++ {
		if v := m.At(row, c); v > bestValue {
			bestValue = v
			bestIndex = c
		}
	}
	return bestIndex
}

func (nn *NeuralNetwork) forward(x *mat.Dense) (*mat.Dense, *mat.Dense) {
	hiddenInput := new(mat.Dense)
	hiddenInput.Mul(x, nn.wHidden)
	addBias(hiddenInput, nn.bHidden)
	hiddenOutput := applySigmoid(hiddenInput)

	outputInput := new(mat.Dense)
	outputInput.Mul(hiddenOutput, nn.wOutput)
	addBias(outputInput, nn.bOutput)
	output := applySigmoid(outputInput)

	return hiddenOutput, output
}

func (nn *NeuralNetwork) Train(x, y *mat.Dense) error {
	if x == nil || y == nil {
		return fmt.Errorf("training inputs and labels must not be nil")
	}
	if nn.config.Loss != "" && nn.config.Loss != "mse" && nn.config.Loss != "bce" {
		return fmt.Errorf("unsupported loss %q", nn.config.Loss)
	}
	if nn.config.Loss == "bce" {
		_, cols := y.Dims()
		if nn.config.OutputNeurons != 1 || cols != 1 {
			return fmt.Errorf("BCE training requires one output and one label column")
		}
		if _, err := BinaryActualLabels(y); err != nil {
			return err
		}
	}
	samples, inputCols := x.Dims()
	labelRows, labelCols := y.Dims()
	if samples == 0 || labelRows == 0 {
		return fmt.Errorf("training data is empty")
	}
	if samples != labelRows {
		return fmt.Errorf("input rows (%d) do not match label rows (%d)", samples, labelRows)
	}
	if inputCols != nn.config.InputNeurons {
		return fmt.Errorf("expected %d input features, got %d", nn.config.InputNeurons, inputCols)
	}
	if labelCols != nn.config.OutputNeurons {
		return fmt.Errorf("expected %d output labels, got %d", nn.config.OutputNeurons, labelCols)
	}

	for epoch := 0; epoch < nn.config.NumEpochs; epoch++ {
		hiddenOutput, predictions := nn.forward(x)

		outputError := subtractDense(predictions, y)
		outputDelta := outputError
		if nn.config.Loss != "bce" {
			outputDelta = elementwiseMultiply(outputError, elementwiseSigmoidPrime(predictions))
		}
		var loss float64
		metric := "mse"
		if nn.config.Loss == "bce" {
			var err error
			loss, err = BinaryCrossEntropy(predictions, y)
			if err != nil {
				return err
			}
			metric = "bce"
		} else {
			loss = MeanSquaredError(predictions, y)
		}

		wOutputT := mat.DenseCopyOf(nn.wOutput.T())
		hiddenError := new(mat.Dense)
		hiddenError.Mul(outputDelta, wOutputT)
		hiddenDelta := elementwiseMultiply(hiddenError, elementwiseSigmoidPrime(hiddenOutput))

		gradWOutput := new(mat.Dense)
		gradWOutput.Mul(hiddenOutput.T(), outputDelta)
		scaleDense(gradWOutput, 1.0/float64(samples))

		gradBOutput := sumAlongAxis0(outputDelta)
		scaleDense(gradBOutput, 1.0/float64(samples))

		gradWHidden := new(mat.Dense)
		gradWHidden.Mul(x.T(), hiddenDelta)
		scaleDense(gradWHidden, 1.0/float64(samples))

		gradBHidden := sumAlongAxis0(hiddenDelta)
		scaleDense(gradBHidden, 1.0/float64(samples))

		subtractScaled(nn.wOutput, gradWOutput, nn.config.LearningRate)
		subtractScaled(nn.bOutput, gradBOutput, nn.config.LearningRate)
		subtractScaled(nn.wHidden, gradWHidden, nn.config.LearningRate)
		subtractScaled(nn.bHidden, gradBHidden, nn.config.LearningRate)

		fmt.Printf("Epoch %d/%d - %s: %.6f\n", epoch+1, nn.config.NumEpochs, metric, loss)

	}

	return nil
}

func (nn *NeuralNetwork) Predict(x *mat.Dense) (*mat.Dense, error) {
	if nn.wHidden == nil || nn.wOutput == nil || nn.bHidden == nil || nn.bOutput == nil {
		return nil, fmt.Errorf("network parameters are not initialized")
	}
	_, output := nn.forward(x)
	return output, nil
}

func Accuracy(predictions, labels *mat.Dense) float64 {
	rows, _ := predictions.Dims()
	var correct int
	for r := 0; r < rows; r++ {
		if argmaxRow(predictions, r) == argmaxRow(labels, r) {
			correct++
		}
	}
	return float64(correct) / float64(rows)
}

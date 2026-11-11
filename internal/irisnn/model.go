package irisnn

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

// NeuralNetwork stores the trainable parameters for a simple
// classifier with one or two hidden layers.
type NeuralNetwork struct {
	adam     *adamState
	config   NeuralNetConfig
	scaler   *FeatureScaler
	wHidden  *mat.Dense
	wHidden2 *mat.Dense
	bHidden2 *mat.Dense
	bHidden  *mat.Dense
	wOutput  *mat.Dense
	bOutput  *mat.Dense
}

type NeuralNetConfig struct {
	Optimizer      string  `json:"optimizer,omitempty"`
	BatchSize      int     `json:"batch_size,omitempty"`
	PositiveWeight float64 `json:"positive_weight,omitempty"`
	HiddenNeurons2 int     `json:"hiddenNeurons2,omitempty"`
	Loss           string  `json:"loss,omitempty"`
	InputNeurons   int     `json:"inputNeurons"`
	OutputNeurons  int     `json:"outputNeurons"`
	HiddenNeurons  int     `json:"hiddenNeurons"`
	NumEpochs      int     `json:"numEpochs"`
	LearningRate   float64 `json:"learningRate"`
	Seed           int     `json:"seed"`
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

	nn := &NeuralNetwork{
		config:  config,
		scaler:  scaler,
		wHidden: randDense(rng, config.InputNeurons, config.HiddenNeurons),
		bHidden: mat.NewDense(1, config.HiddenNeurons, nil),
		wOutput: randDense(rng, config.HiddenNeurons, config.OutputNeurons),
		bOutput: mat.NewDense(1, config.OutputNeurons, nil),
	}
	if config.HiddenNeurons2 > 0 {
		nn.wHidden = heDense(rng, config.InputNeurons, config.HiddenNeurons)
		nn.wHidden2 = heDense(rng, config.HiddenNeurons, config.HiddenNeurons2)
		nn.bHidden2 = mat.NewDense(1, config.HiddenNeurons2, nil)
		nn.wOutput = randDense(rng, config.HiddenNeurons2, config.OutputNeurons)
	}
	return nn
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

func heDense(rng *rand.Rand, rows, cols int) *mat.Dense {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = rng.NormFloat64() * math.Sqrt(2/float64(rows))
	}
	return mat.NewDense(rows, cols, data)
}

func relu(m *mat.Dense) *mat.Dense {
	r, c := m.Dims()
	out := mat.NewDense(r, c, nil)
	out.Apply(func(_, _ int, v float64) float64 { return math.Max(0, v) }, m)
	return out
}

// A positive ReLU activation is equivalent to a positive pre-activation.
func reluDerivative(m *mat.Dense) *mat.Dense {
	r, c := m.Dims()
	out := mat.NewDense(r, c, nil)
	out.Apply(func(_, _ int, v float64) float64 {
		if v > 0 {
			return 1
		}
		return 0
	}, m)
	return out
}

func (nn *NeuralNetwork) forwardLayers(x *mat.Dense) (*mat.Dense, *mat.Dense, *mat.Dense) {
	z := new(mat.Dense)
	z.Mul(x, nn.wHidden)
	addBias(z, nn.bHidden)
	var h1, h2 *mat.Dense
	if nn.config.HiddenNeurons2 > 0 {
		h1 = relu(z)
		z2 := new(mat.Dense)
		z2.Mul(h1, nn.wHidden2)
		addBias(z2, nn.bHidden2)
		h2 = relu(z2)
	} else {
		h1 = applySigmoid(z)
		h2 = h1
	}
	out := new(mat.Dense)
	out.Mul(h2, nn.wOutput)
	addBias(out, nn.bOutput)
	return h1, h2, applySigmoid(out)
}

func (nn *NeuralNetwork) forward(x *mat.Dense) (*mat.Dense, *mat.Dense) {
	_, hidden, out := nn.forwardLayers(x)
	return hidden, out
}

// PositiveWeight treats an omitted legacy configuration value as ordinary BCE.
func (nn *NeuralNetwork) PositiveWeight() float64 {
	if nn.config.PositiveWeight == 0 {
		return 1
	}
	return nn.config.PositiveWeight
}

func (nn *NeuralNetwork) Train(x, y *mat.Dense) error {
	return nn.TrainWithEpochCallback(x, y, nil)
}

// TrainWithEpochCallback calls callback after each completed epoch (numbered from 1).
func (nn *NeuralNetwork) TrainWithEpochCallback(x, y *mat.Dense, callback func(int) error) error {
	if err := validateOptimizer(nn.config.Optimizer); err != nil {
		return err
	}
	if nn.config.BatchSize < 0 {
		return fmt.Errorf("batch size cannot be negative")
	}

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
	if w := nn.PositiveWeight(); w <= 0 || math.IsNaN(w) || math.IsInf(w, 0) {
		return fmt.Errorf("positive weight must be finite and greater than zero")
	}
	if nn.config.HiddenNeurons2 < 0 {
		return fmt.Errorf("second hidden width cannot be negative")
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

	rng := rand.New(rand.NewSource(int64(nn.config.Seed)))
	for epoch := 0; epoch < nn.config.NumEpochs; epoch++ {
		if nn.config.BatchSize == 0 {
			if err := nn.trainBatch(x, y); err != nil {
				return err
			}
		} else {
			order := rng.Perm(samples)
			for start := 0; start < samples; start += nn.config.BatchSize {
				end := start + min(nn.config.BatchSize, samples-start)
				bx := mat.NewDense(end-start, inputCols, nil)
				by := mat.NewDense(end-start, labelCols, nil)
				for row, original := range order[start:end] {
					bx.SetRow(row, x.RawRowView(original))
					by.SetRow(row, y.RawRowView(original))
				}
				if err := nn.trainBatch(bx, by); err != nil {
					return err
				}
			}
		}
		if callback != nil {
			if err := callback(epoch + 1); err != nil {
				return err
			}
			continue
		}
		predictions, err := nn.Predict(x)
		if err != nil {
			return err
		}
		loss := MeanSquaredError(predictions, y)
		metric := "mse"
		if nn.config.Loss == "bce" {
			loss, err = WeightedBinaryCrossEntropy(predictions, y, nn.PositiveWeight())
			metric = "weighted_bce"
			if err != nil {
				return err
			}
		}
		fmt.Printf("Epoch %d/%d - %s: %.6f\n", epoch+1, nn.config.NumEpochs, metric, loss)
	}
	return nil
}

func (nn *NeuralNetwork) trainBatch(x, y *mat.Dense) error {
	samples, _ := x.Dims()

	hiddenFirst, hiddenOutput, predictions := nn.forwardLayers(x)

	outputError := subtractDense(predictions, y)
	outputDelta := outputError
	if nn.config.Loss != "bce" {
		outputDelta = elementwiseMultiply(outputError, elementwiseSigmoidPrime(predictions))
	}
	if nn.config.Loss == "bce" {
		for row := 0; row < samples; row++ {
			if y.At(row, 0) == 1 {
				outputDelta.Set(row, 0, outputDelta.At(row, 0)*nn.PositiveWeight())
			}
		}
	}

	wOutputT := mat.DenseCopyOf(nn.wOutput.T())
	hiddenError := new(mat.Dense)
	hiddenError.Mul(outputDelta, wOutputT)
	hiddenDelta := elementwiseMultiply(hiddenError, elementwiseSigmoidPrime(hiddenOutput))
	var gradWHidden2, gradBHidden2 *mat.Dense
	if nn.config.HiddenNeurons2 > 0 {
		delta2 := elementwiseMultiply(hiddenError, reluDerivative(hiddenOutput))
		gradWHidden2 = new(mat.Dense)
		gradWHidden2.Mul(hiddenFirst.T(), delta2)
		scaleDense(gradWHidden2, 1/float64(samples))
		gradBHidden2 = sumAlongAxis0(delta2)
		scaleDense(gradBHidden2, 1/float64(samples))
		error1 := new(mat.Dense)
		error1.Mul(delta2, nn.wHidden2.T())
		hiddenDelta = elementwiseMultiply(error1, reluDerivative(hiddenFirst))
	}

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

	parameters := []*mat.Dense{nn.wHidden, nn.bHidden, nn.wOutput, nn.bOutput}
	gradients := []*mat.Dense{gradWHidden, gradBHidden, gradWOutput, gradBOutput}
	if nn.config.HiddenNeurons2 > 0 {
		parameters = append(parameters, nn.wHidden2, nn.bHidden2)
		gradients = append(gradients, gradWHidden2, gradBHidden2)
	}
	nn.updateParameters(parameters, gradients)

	return nil
}

func (nn *NeuralNetwork) Predict(x *mat.Dense) (*mat.Dense, error) {
	if x == nil {
		return nil, fmt.Errorf("prediction inputs are nil")
	}
	_, cols := x.Dims()
	if cols != nn.config.InputNeurons {
		return nil, fmt.Errorf("expected %d input features, got %d", nn.config.InputNeurons, cols)
	}
	if nn.config.HiddenNeurons2 > 0 && (nn.wHidden2 == nil || nn.bHidden2 == nil) {
		return nil, fmt.Errorf("second hidden layer is not initialized")
	}

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

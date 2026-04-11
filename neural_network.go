package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	"gonum.org/v1/gonum/mat"
)

// NeuralNetwork stores the trainable parameters for a simple
// single-hidden-layer classifier.
type NeuralNetwork struct {
	config  NeuralNetConfig
	wHidden *mat.Dense
	bHidden *mat.Dense
	wOutput *mat.Dense
	bOutput *mat.Dense
}

type NeuralNetConfig struct {
	inputNeurons  int
	outputNeurons int
	hiddenNeurons int
	numEpochs     int
	learningRate  float64
}

// dataset keeps one split of the Iris data in matrix form.
type dataset struct {
	inputs *mat.Dense
	labels *mat.Dense
	rows   int
}

// featureScaler stores the statistics used to normalize every feature column
// using the training set only.
type featureScaler struct {
	mean []float64
	std  []float64
}

func newNeuralNetwork(config NeuralNetConfig) *NeuralNetwork {
	// Start with small random weights and zero biases.
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	return &NeuralNetwork{
		config:  config,
		wHidden: randDense(rng, config.inputNeurons, config.hiddenNeurons),
		bHidden: mat.NewDense(1, config.hiddenNeurons, nil),
		wOutput: randDense(rng, config.hiddenNeurons, config.outputNeurons),
		bOutput: mat.NewDense(1, config.outputNeurons, nil),
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
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidPrimeFromActivation(a float64) float64 {
	// The derivative is written in terms of the activation value so we do not
	// need to recalculate sigmoid(z) during backpropagation.
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
	// Bias vectors are stored as 1 x N matrices, so we add the same bias row
	// to every sample in the batch.
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

func meanSquaredError(predictions, labels *mat.Dense) float64 {
	// Mean squared error is enough for this small educational classifier and
	// gives a readable training loss during optimization.
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
	// Sum each output column across all rows. This is used to build the bias
	// gradient during backpropagation.
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

func (nn *NeuralNetwork) forward(x *mat.Dense) (*mat.Dense, *mat.Dense) {
	// Forward propagation:
	// input -> hidden linear transform -> sigmoid
	// hidden -> output linear transform -> sigmoid
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

func (nn *NeuralNetwork) train(x, y *mat.Dense) error {
	// Validate that the input data shape matches the network configuration
	// before starting gradient descent.
	samples, inputCols := x.Dims()
	labelRows, labelCols := y.Dims()
	if samples == 0 || labelRows == 0 {
		return fmt.Errorf("training data is empty")
	}
	if samples != labelRows {
		return fmt.Errorf("input rows (%d) do not match label rows (%d)", samples, labelRows)
	}
	if inputCols != nn.config.inputNeurons {
		return fmt.Errorf("expected %d input features, got %d", nn.config.inputNeurons, inputCols)
	}
	if labelCols != nn.config.outputNeurons {
		return fmt.Errorf("expected %d output labels, got %d", nn.config.outputNeurons, labelCols)
	}

	for epoch := 0; epoch < nn.config.numEpochs; epoch++ {
		hiddenOutput, predictions := nn.forward(x)

		// Backpropagation starts at the output layer by comparing predictions
		// against the expected one-hot encoded labels.
		outputError := subtractDense(predictions, y)
		outputDelta := elementwiseMultiply(outputError, elementwiseSigmoidPrime(predictions))

		// Propagate the output error back into the hidden layer.
		wOutputT := mat.DenseCopyOf(nn.wOutput.T())
		hiddenError := new(mat.Dense)
		hiddenError.Mul(outputDelta, wOutputT)
		hiddenDelta := elementwiseMultiply(hiddenError, elementwiseSigmoidPrime(hiddenOutput))

		// Build gradients for weights and biases, average them across the batch,
		// then apply a gradient descent step.
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

		subtractScaled(nn.wOutput, gradWOutput, nn.config.learningRate)
		subtractScaled(nn.bOutput, gradBOutput, nn.config.learningRate)
		subtractScaled(nn.wHidden, gradWHidden, nn.config.learningRate)
		subtractScaled(nn.bHidden, gradBHidden, nn.config.learningRate)

		if epoch == 0 || (epoch+1)%500 == 0 || epoch == nn.config.numEpochs-1 {
			loss := meanSquaredError(predictions, y)
			fmt.Printf("Epoch %d/%d - loss: %.6f\n", epoch+1, nn.config.numEpochs, loss)
		}
	}

	return nil
}

func (nn *NeuralNetwork) predict(x *mat.Dense) (*mat.Dense, error) {
	if nn.wHidden == nil || nn.wOutput == nil || nn.bHidden == nil || nn.bOutput == nil {
		return nil, fmt.Errorf("network parameters are not initialized")
	}
	// Prediction is just a forward pass after training.
	_, output := nn.forward(x)
	return output, nil
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

func accuracy(predictions, labels *mat.Dense) float64 {
	rows, _ := predictions.Dims()
	var correct int
	for r := 0; r < rows; r++ {
		if argmaxRow(predictions, r) == argmaxRow(labels, r) {
			correct++
		}
	}
	return float64(correct) / float64(rows)
}

func loadDataset(path string) (*dataset, error) {
	// The split files contain:
	// Id, 4 numeric features, and 3 one-hot output columns.
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	if len(records) < 2 {
		return nil, fmt.Errorf("%s must contain a header and at least one data row", path)
	}

	samples := len(records) - 1
	inputsData := make([]float64, samples*4)
	labelsData := make([]float64, samples*3)

	inputIndex := 0
	labelIndex := 0
	for rowIndex, record := range records[1:] {
		if len(record) != 8 {
			return nil, fmt.Errorf("%s row %d has %d columns, expected 8", path, rowIndex+2, len(record))
		}

		for _, field := range record[1:5] {
			value, err := strconv.ParseFloat(field, 64)
			if err != nil {
				return nil, fmt.Errorf("%s row %d feature parse error: %w", path, rowIndex+2, err)
			}
			inputsData[inputIndex] = value
			inputIndex++
		}

		for _, field := range record[5:8] {
			value, err := strconv.ParseFloat(field, 64)
			if err != nil {
				return nil, fmt.Errorf("%s row %d label parse error: %w", path, rowIndex+2, err)
			}
			labelsData[labelIndex] = value
			labelIndex++
		}
	}

	return &dataset{
		inputs: mat.NewDense(samples, 4, inputsData),
		labels: mat.NewDense(samples, 3, labelsData),
		rows:   samples,
	}, nil
}

func fitFeatureScaler(inputs *mat.Dense) *featureScaler {
	// Normalize each feature column to roughly zero mean and unit variance.
	// This makes gradient descent converge more reliably.
	rows, cols := inputs.Dims()
	mean := make([]float64, cols)
	std := make([]float64, cols)

	for c := 0; c < cols; c++ {
		var sum float64
		for r := 0; r < rows; r++ {
			sum += inputs.At(r, c)
		}
		mean[c] = sum / float64(rows)

		var variance float64
		for r := 0; r < rows; r++ {
			diff := inputs.At(r, c) - mean[c]
			variance += diff * diff
		}

		std[c] = math.Sqrt(variance / float64(rows))
		if std[c] == 0 {
			std[c] = 1
		}
	}

	return &featureScaler{mean: mean, std: std}
}

func (s *featureScaler) transform(inputs *mat.Dense) *mat.Dense {
	rows, cols := inputs.Dims()
	out := mat.NewDense(rows, cols, nil)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			out.Set(r, c, (inputs.At(r, c)-s.mean[c])/s.std[c])
		}
	}
	return out
}

func evaluateSplit(name string, network *NeuralNetwork, data *dataset) {
	// Evaluate each dataset split with both loss and classification accuracy.
	predictions, err := network.predict(data.inputs)
	if err != nil {
		log.Fatalf("predict %s: %v", name, err)
	}

	fmt.Printf(
		"%s - rows: %d, loss: %.6f, accuracy: %.2f%%\n",
		name,
		data.rows,
		meanSquaredError(predictions, data.labels),
		accuracy(predictions, data.labels)*100,
	)
}

func main() {
	// Load the preprocessed CSV files generated by train_test_split.go.
	trainSet, err := loadDataset("train.csv")
	if err != nil {
		log.Fatal(err)
	}
	validationSet, err := loadDataset("validation.csv")
	if err != nil {
		log.Fatal(err)
	}
	testSet, err := loadDataset("test.csv")
	if err != nil {
		log.Fatal(err)
	}

	scaler := fitFeatureScaler(trainSet.inputs)
	trainSet.inputs = scaler.transform(trainSet.inputs)
	validationSet.inputs = scaler.transform(validationSet.inputs)
	testSet.inputs = scaler.transform(testSet.inputs)

	// A small hidden layer is enough for the Iris dataset.
	config := NeuralNetConfig{
		inputNeurons:  4,
		outputNeurons: 3,
		hiddenNeurons: 8,
		numEpochs:     4000,
		learningRate:  0.15,
	}

	network := newNeuralNetwork(config)
	if err := network.train(trainSet.inputs, trainSet.labels); err != nil {
		log.Fatal(err)
	}

	// Report performance on all three splits so training quality is easy to inspect.
	evaluateSplit("Train", network, trainSet)
	evaluateSplit("Validation", network, validationSet)
	evaluateSplit("Test", network, testSet)
}

//go:build ignore
// +build ignore

package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"go_neural_network/internal/irisnn"
	"gonum.org/v1/gonum/mat"
)

const (
	numInputFeatures = 14
	numHiddenNeurons = 32
	numOutputNeurons = 1 // P(fraud)
	expectedFieldsNN = 10
	defaultCutoffNN  = 0.5
)

var txTypes = []string{"PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"}

// FraudDataset holds one split of fraud data in matrix form
type FraudDataset struct {
	Inputs *mat.Dense // features
	Labels *mat.Dense // binary: [0] or [1]
	Rows   int
}

// NNMetrics tracks performance metrics for neural network models
type NNMetrics struct {
	Loss      float64 `json:"loss"` // BCE loss
	Accuracy  float64 `json:"accuracy"`
	Precision float64 `json:"precision"`
	Recall    float64 `json:"recall"`
	F1        float64 `json:"f1"`
	F2        float64 `json:"f2"`
	PosRate   float64 `json:"positive_rate"`
	Rows      int64   `json:"rows"`
	TruePos   int64   `json:"true_pos"`
	FalsePos  int64   `json:"false_pos"`
	TrueNeg   int64   `json:"true_neg"`
	FalseNeg  int64   `json:"false_neg"`
}

// FraudNeuralNetwork implements single-hidden-layer classifier for fraud
type FraudNeuralNetwork struct {
	Config       NNConfig       `json:"config"`
	Scaler       *FeatureScaler `json:"scaler"`
	WeightsHid   *mat.Dense     `json:"-"`
	BiasHid      *mat.Dense     `json:"-"`
	WeightsOut   *mat.Dense     `json:"-"`
	BiasOut      *mat.Dense     `json:"-"`
	FeatureNames []string       `json:"feature_names"`
	TypeIndex    map[string]int `json:"type_index"`
	Threshold    float64        `json:"threshold"`
	Summary      TrainSummaryNN `json:"summary"`
}

// NNConfig holds neural network hyperparameters
type NNConfig struct {
	InputNeurons  int     `json:"input_neurons"`
	HiddenNeurons int     `json:"hidden_neurons"`
	OutputNeurons int     `json:"output_neurons"`
	NumEpochs     int     `json:"epochs"`
	LearningRate  float64 `json:"learning_rate"`
	L2Regularizer float64 `json:"l2_regularizer"`
	Seed          int64   `json:"seed"`
}

// FeatureScaler normalizes features to zero mean and unit variance
type FeatureScaler struct {
	Mean []float64 `json:"mean"`
	Std  []float64 `json:"std"`
}

// TrainSummaryNN tracks training results
type TrainSummaryNN struct {
	Train      NNMetrics     `json:"train"`
	Validation NNMetrics     `json:"validation"`
	Test       NNMetrics     `json:"test"`
	Epochs     int           `json:"epochs"`
	Elapsed    time.Duration `json:"elapsed"`
	LearnedAt  time.Time     `json:"learned_at"`
	Config     NNConfig      `json:"config"`
	Threshold  float64       `json:"threshold"`
}

// FraudTransaction represents one transaction record
type FraudTransaction struct {
	Step           float64
	Type           string
	Amount         float64
	OldBalanceOrig float64
	NewBalanceOrig float64
	NameDest       string
	OldBalanceDest float64
	NewBalanceDest float64
	IsFraud        int
}

// NetworkWeights holds serializable weight data for model persistence
type NetworkWeights struct {
	WeightsHid []float64 `json:"weights_hidden"`
	BiasHid    []float64 `json:"bias_hidden"`
	WeightsOut []float64 `json:"weights_output"`
	BiasOut    []float64 `json:"bias_output"`
	HidShape   [2]int    `json:"hidden_shape"`
	OutShape   [2]int    `json:"output_shape"`
}

// FraudModelArtifact holds complete trained model for persistence
type FraudModelArtifact struct {
	Weights      NetworkWeights `json:"weights"`
	Config       NNConfig       `json:"config"`
	Scaler       *FeatureScaler `json:"scaler"`
	FeatureNames []string       `json:"feature_names"`
	TypeIndex    map[string]int `json:"type_index"`
	Threshold    float64        `json:"threshold"`
	Summary      TrainSummaryNN `json:"summary"`
}

func main() {
	mode := flag.String("mode", "train", "train, predict-csv, or predict-one")
	trainInput := flag.String("train-input", "credit-card-fraud-datasets/transactions_train_split.csv", "Path to training CSV")
	validationInput := flag.String("validation-input", "credit-card-fraud-datasets/transactions_validation.csv", "Path to validation CSV")
	testInput := flag.String("test-input", "credit-card-fraud-datasets/transactions_test.csv", "Path to test CSV")
	inputPath := flag.String("input", "credit-card-fraud-datasets/transactions_train.csv", "Path to prediction input CSV")
	modelPath := flag.String("model", "credit-card-fraud-datasets/transactions_model_nn.json", "Path to save/load neural network model")
	outputPath := flag.String("output", "credit-card-fraud-datasets/transactions_predictions_nn.csv", "Path to write predictions")
	epochs := flag.Int("epochs", 10, "Number of training epochs")
	learningRate := flag.Float64("learning-rate", 0.01, "Learning rate")
	l2 := flag.Float64("l2", 0.00001, "L2 regularization strength")
	hiddenNeurons := flag.Int("hidden-neurons", numHiddenNeurons, "Number of hidden layer neurons")
	maxRows := flag.Int("max-rows", 0, "Maximum rows to process; 0 for all")
	thresholdStep := flag.Float64("threshold-step", 0.01, "Threshold search step")
	workers := flag.Int("workers", runtime.NumCPU(), "Worker goroutines for prediction")

	// Single prediction flags
	step := flag.Float64("step", 0, "Transaction step")
	txType := flag.String("type", "", "Transaction type")
	amount := flag.Float64("amount", 0, "Transaction amount")
	oldBalanceOrig := flag.Float64("oldbalance-orig", 0, "Origin old balance")
	newBalanceOrig := flag.Float64("newbalance-orig", 0, "Origin new balance")
	nameDest := flag.String("name-dest", "", "Destination account id")
	oldBalanceDest := flag.Float64("oldbalance-dest", 0, "Destination old balance")
	newBalanceDest := flag.Float64("newbalance-dest", 0, "Destination new balance")
	flag.Parse()

	switch *mode {
	case "train":
		model, err := trainFraudNeuralNetwork(*trainInput, *validationInput, *testInput,
			*epochs, *learningRate, *l2, *hiddenNeurons, *maxRows, *thresholdStep)
		if err != nil {
			log.Fatal(err)
		}
		if err := saveFraudModelNN(*modelPath, model); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("Saved neural network model to %s\n", *modelPath)

	case "predict-csv":
		model, err := loadFraudModelNN(*modelPath)
		if err != nil {
			log.Fatal(err)
		}
		if *workers < 1 {
			log.Fatal("workers must be at least 1")
		}
		if err := predictCSVNN(model, *inputPath, *outputPath, *maxRows, *workers); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("Wrote predictions to %s\n", *outputPath)
		fmt.Printf("Processed with %d worker goroutines\n", *workers)

	case "predict-one":
		model, err := loadFraudModelNN(*modelPath)
		if err != nil {
			log.Fatal(err)
		}
		tx := FraudTransaction{
			Step:           *step,
			Type:           *txType,
			Amount:         *amount,
			OldBalanceOrig: *oldBalanceOrig,
			NewBalanceOrig: *newBalanceOrig,
			NameDest:       *nameDest,
			OldBalanceDest: *oldBalanceDest,
			NewBalanceDest: *newBalanceDest,
		}
		probability, label, err := predictTransactionNN(model, tx)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("Fraud probability: %.6f\n", probability)
		fmt.Printf("Predicted label: %d\n", label)
		fmt.Printf("Threshold used: %.2f\n", model.Threshold)

	default:
		log.Fatalf("unsupported mode: %q", *mode)
	}
}

// trainFraudNeuralNetwork trains a neural network on fraud data
func trainFraudNeuralNetwork(trainPath, validPath, testPath string, epochs int,
	learningRate, l2 float64, hiddenNeurons, maxRows int, thresholdStep float64) (*FraudNeuralNetwork, error) {

	start := time.Now()

	// Load datasets into matrices
	trainData, err := loadFraudDataset(trainPath, maxRows)
	if err != nil {
		return nil, fmt.Errorf("load train data: %w", err)
	}

	validData, err := loadFraudDataset(validPath, maxRows)
	if err != nil {
		return nil, fmt.Errorf("load validation data: %w", err)
	}

	testData, err := loadFraudDataset(testPath, maxRows)
	if err != nil {
		return nil, fmt.Errorf("load test data: %w", err)
	}

	// Fit feature scaler on training data
	scaler := fitFeatureScaler(trainData.Inputs)

	// Normalize all datasets
	trainInputs := scaler.transform(trainData.Inputs)
	validInputs := scaler.transform(validData.Inputs)
	testInputs := scaler.transform(testData.Inputs)

	// Initialize neural network
	config := NNConfig{
		InputNeurons:  numInputFeatures,
		HiddenNeurons: hiddenNeurons,
		OutputNeurons: numOutputNeurons,
		NumEpochs:     epochs,
		LearningRate:  learningRate,
		L2Regularizer: l2,
		Seed:          time.Now().UnixNano(),
	}

	model := newFraudNeuralNetwork(config, scaler)

	// Training loop
	for epoch := 0; epoch < epochs; epoch++ {
		// Forward pass and backpropagation
		hiddenOut, predictions := model.forward(trainInputs)
		model.backward(trainInputs, hiddenOut, predictions, trainData.Labels, l2)

		// Tune threshold on validation set
		threshold, valMetrics, err := tuneThresholdNN(model, validInputs, validData.Labels, thresholdStep)
		if err != nil {
			return nil, err
		}
		model.Threshold = threshold

		// Evaluate on all splits
		trainMetrics, err := evaluateNN(model, trainInputs, trainData.Labels, threshold)
		if err != nil {
			return nil, err
		}
		valMetrics, err = evaluateNN(model, validInputs, validData.Labels, threshold)
		if err != nil {
			return nil, err
		}
		testMetrics, err := evaluateNN(model, testInputs, testData.Labels, threshold)
		if err != nil {
			return nil, err
		}

		fmt.Printf(
			"Epoch %d/%d - train: bce=%.6f acc=%.4f precision=%.4f recall=%.4f f2=%.4f | "+
				"val: bce=%.6f acc=%.4f precision=%.4f recall=%.4f f2=%.4f | "+
				"test: bce=%.6f acc=%.4f precision=%.4f recall=%.4f f2=%.4f | threshold=%.2f\n",
			epoch+1, epochs,
			trainMetrics.Loss, trainMetrics.Accuracy, trainMetrics.Precision, trainMetrics.Recall, trainMetrics.F2,
			valMetrics.Loss, valMetrics.Accuracy, valMetrics.Precision, valMetrics.Recall, valMetrics.F2,
			testMetrics.Loss, testMetrics.Accuracy, testMetrics.Precision, testMetrics.Recall, testMetrics.F2,
			threshold,
		)

		model.Summary = TrainSummaryNN{
			Train:      trainMetrics,
			Validation: valMetrics,
			Test:       testMetrics,
			Epochs:     epoch + 1,
			Elapsed:    time.Since(start),
			LearnedAt:  time.Now(),
			Config:     config,
			Threshold:  threshold,
		}
	}

	return model, nil
}

// newFraudNeuralNetwork creates and initializes a neural network
func newFraudNeuralNetwork(config NNConfig, scaler *FeatureScaler) *FraudNeuralNetwork {
	rng := rand.New(rand.NewSource(config.Seed))

	nn := &FraudNeuralNetwork{
		Config:       config,
		Scaler:       scaler,
		WeightsHid:   randDenseMatrix(rng, config.InputNeurons, config.HiddenNeurons),
		BiasHid:      mat.NewDense(1, config.HiddenNeurons, nil),
		WeightsOut:   randDenseMatrix(rng, config.HiddenNeurons, config.OutputNeurons),
		BiasOut:      mat.NewDense(1, config.OutputNeurons, nil),
		FeatureNames: fraudFeatureNames(),
		TypeIndex:    buildTypeIndex(),
		Threshold:    defaultCutoffNN,
	}

	return nn
}

// randDenseMatrix creates a matrix with small random values
func randDenseMatrix(rng *rand.Rand, rows, cols int) *mat.Dense {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = (rng.Float64()*2 - 1) * 0.5
	}
	return mat.NewDense(rows, cols, data)
}

// forward performs forward propagation
func (nn *FraudNeuralNetwork) forward(x *mat.Dense) (*mat.Dense, *mat.Dense) {
	// Hidden layer: Z = X * W_h + b_h
	hiddenZ := new(mat.Dense)
	hiddenZ.Mul(x, nn.WeightsHid)
	addMatrixBias(hiddenZ, nn.BiasHid)
	hiddenOut := applySigmoidMatrix(hiddenZ)

	// Output layer: Z = H * W_o + b_o
	outputZ := new(mat.Dense)
	outputZ.Mul(hiddenOut, nn.WeightsOut)
	addMatrixBias(outputZ, nn.BiasOut)
	outputPred := applySigmoidMatrix(outputZ)

	return hiddenOut, outputPred
}

// backward performs backpropagation with gradient updates
func (nn *FraudNeuralNetwork) backward(x, hiddenOut, predictions, labels *mat.Dense, l2 float64) {
	samples, _ := x.Dims()
	sampleFloat := float64(samples)

	// Output layer error and delta
	outDelta := subtractMatrices(predictions, labels)

	// Backpropagate to hidden layer
	hidWeightsT := mat.DenseCopyOf(nn.WeightsOut.T())
	hidError := new(mat.Dense)
	hidError.Mul(outDelta, hidWeightsT)
	hidDelta := multiplyElementwise(hidError, sigmoidPrimeMatrix(hiddenOut))

	// Compute gradients
	gradWeightsOut := new(mat.Dense)
	gradWeightsOut.Mul(hiddenOut.T(), outDelta)
	scaleMatrix(gradWeightsOut, 1.0/sampleFloat)

	gradBiasOut := sumAxis0(outDelta)
	scaleMatrix(gradBiasOut, 1.0/sampleFloat)

	gradWeightsHid := new(mat.Dense)
	gradWeightsHid.Mul(x.T(), hidDelta)
	scaleMatrix(gradWeightsHid, 1.0/sampleFloat)

	gradBiasHid := sumAxis0(hidDelta)
	scaleMatrix(gradBiasHid, 1.0/sampleFloat)

	// Apply L2 regularization to weights
	regOut := mat.DenseCopyOf(nn.WeightsOut)
	scaleMatrix(regOut, l2)
	addMatrices(gradWeightsOut, regOut)

	regHid := mat.DenseCopyOf(nn.WeightsHid)
	scaleMatrix(regHid, l2)
	addMatrices(gradWeightsHid, regHid)

	// Gradient descent update
	subtractScaledMatrix(nn.WeightsOut, gradWeightsOut, nn.Config.LearningRate)
	subtractScaledMatrix(nn.BiasOut, gradBiasOut, nn.Config.LearningRate)
	subtractScaledMatrix(nn.WeightsHid, gradWeightsHid, nn.Config.LearningRate)
	subtractScaledMatrix(nn.BiasHid, gradBiasHid, nn.Config.LearningRate)
}

// Helper matrix operations

func applySigmoidMatrix(m *mat.Dense) *mat.Dense {
	rows, cols := m.Dims()
	out := mat.NewDense(rows, cols, nil)
	out.Apply(func(_, _ int, v float64) float64 {
		return sigmoid(v)
	}, m)
	return out
}

func sigmoidPrimeMatrix(m *mat.Dense) *mat.Dense {
	rows, cols := m.Dims()
	out := mat.NewDense(rows, cols, nil)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			val := m.At(r, c)
			out.Set(r, c, val*(1.0-val))
		}
	}
	return out
}

func sigmoid(x float64) float64 {
	if x >= 0 {
		return 1.0 / (1.0 + math.Exp(-x))
	}
	z := math.Exp(x)
	return z / (1.0 + z)
}

func multiplyElementwise(a, b *mat.Dense) *mat.Dense {
	rows, cols := a.Dims()
	out := mat.NewDense(rows, cols, nil)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			out.Set(r, c, a.At(r, c)*b.At(r, c))
		}
	}
	return out
}

func subtractMatrices(a, b *mat.Dense) *mat.Dense {
	rows, cols := a.Dims()
	out := mat.NewDense(rows, cols, nil)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			out.Set(r, c, a.At(r, c)-b.At(r, c))
		}
	}
	return out
}

func addMatrices(dst, src *mat.Dense) {
	rows, cols := dst.Dims()
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			dst.Set(r, c, dst.At(r, c)+src.At(r, c))
		}
	}
}

func addMatrixBias(m, bias *mat.Dense) {
	rows, cols := m.Dims()
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			m.Set(r, c, m.At(r, c)+bias.At(0, c))
		}
	}
}

func subtractScaledMatrix(dst, src *mat.Dense, scale float64) {
	rows, cols := dst.Dims()
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			dst.Set(r, c, dst.At(r, c)-scale*src.At(r, c))
		}
	}
}

func scaleMatrix(m *mat.Dense, factor float64) {
	rows, cols := m.Dims()
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			m.Set(r, c, m.At(r, c)*factor)
		}
	}
}

func sumAxis0(m *mat.Dense) *mat.Dense {
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

// loadFraudDataset loads CSV data into matrices with binary labels
func loadFraudDataset(path string, maxRows int) (*FraudDataset, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open file: %w", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	if _, err := reader.Read(); err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}

	var records [][]string
	for {
		row, err := reader.Read()
		if err != nil {
			if err == io.EOF {
				break
			}
			return nil, fmt.Errorf("read row: %w", err)
		}
		if maxRows > 0 && len(records) >= maxRows {
			break
		}
		records = append(records, row)
	}

	if len(records) == 0 {
		return nil, fmt.Errorf("no data rows in %s", path)
	}

	inputsData := make([]float64, len(records)*numInputFeatures)
	labelsData := make([]float64, len(records)*numOutputNeurons)

	inputIdx := 0
	labelIdx := 0
	for _, row := range records {
		tx, err := parseTransactionRecord(row)
		if err != nil {
			return nil, fmt.Errorf("parse record: %w", err)
		}

		features := fraudNumericFeatures(tx)
		for _, f := range features {
			inputsData[inputIdx] = f
			inputIdx++
		}

		if tx.IsFraud != 0 && tx.IsFraud != 1 {
			return nil, fmt.Errorf("invalid binary label %d", tx.IsFraud)
		}
		labelsData[labelIdx] = float64(tx.IsFraud)
		labelIdx++
	}

	return &FraudDataset{
		Inputs: mat.NewDense(len(records), numInputFeatures, inputsData),
		Labels: mat.NewDense(len(records), numOutputNeurons, labelsData),
		Rows:   len(records),
	}, nil
}

// fitFeatureScaler computes mean and std from the input matrix
func fitFeatureScaler(inputs *mat.Dense) *FeatureScaler {
	rows, cols := inputs.Dims()
	mean := make([]float64, cols)
	std := make([]float64, cols)

	// Compute means
	for c := 0; c < cols; c++ {
		var sum float64
		for r := 0; r < rows; r++ {
			sum += inputs.At(r, c)
		}
		mean[c] = sum / float64(rows)
	}

	// Compute standard deviations
	for c := 0; c < cols; c++ {
		var variance float64
		for r := 0; r < rows; r++ {
			diff := inputs.At(r, c) - mean[c]
			variance += diff * diff
		}
		std[c] = math.Sqrt(variance / float64(rows))
		if std[c] == 0 {
			std[c] = 1.0
		}
	}

	return &FeatureScaler{Mean: mean, Std: std}
}

// transform normalizes a matrix using fitted statistics
func (fs *FeatureScaler) transform(inputs *mat.Dense) *mat.Dense {
	rows, cols := inputs.Dims()
	out := mat.NewDense(rows, cols, nil)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			normalized := (inputs.At(r, c) - fs.Mean[c]) / fs.Std[c]
			out.Set(r, c, normalized)
		}
	}
	return out
}

// tuneThresholdNN finds the best threshold on validation set for F2 metric
func tuneThresholdNN(model *FraudNeuralNetwork, validInputs, validLabels *mat.Dense, step float64) (float64, NNMetrics, error) {
	if step <= 0 || step >= 1 {
		step = 0.01
	}

	predictions, err := model.predict(validInputs)
	if err != nil {
		return 0, NNMetrics{}, err
	}

	fraudProbs, err := irisnn.BinaryPositiveClassProbabilities(predictions)
	if err != nil {
		return 0, NNMetrics{}, err
	}
	fraudActuals, err := irisnn.BinaryActualLabels(validLabels)
	if err != nil {
		return 0, NNMetrics{}, err
	}

	bestThreshold := defaultCutoffNN
	bestMetrics := NNMetrics{F2: -1}

	for threshold := step; threshold < 1.0; threshold += step {
		metrics, err := evaluateThreshold(fraudProbs, fraudActuals, threshold)
		if err != nil {
			return 0, NNMetrics{}, err
		}
		if isBetterThreshold(metrics, bestMetrics) {
			bestThreshold = threshold
			bestMetrics = metrics
		}
	}

	if bestMetrics.F2 < 0 {
		bestMetrics, err = evaluateThreshold(fraudProbs, fraudActuals, defaultCutoffNN)
		if err != nil {
			return 0, NNMetrics{}, err
		}
		bestThreshold = defaultCutoffNN
	}

	return bestThreshold, bestMetrics, nil
}

// evaluateThreshold computes metrics at a specific threshold
func evaluateThreshold(probs []float64, actuals []int, threshold float64) (NNMetrics, error) {
	m, err := irisnn.EvaluateBinaryThreshold(probs, actuals, threshold)
	return nnMetrics(m), err
}

func nnMetrics(m irisnn.BinaryMetrics) NNMetrics {
	return NNMetrics{Loss: m.Loss, Accuracy: m.Accuracy, Precision: m.Precision, Recall: m.Recall,
		F1: m.F1, F2: m.F2, PosRate: m.PosRate, Rows: m.Rows, TruePos: m.TruePos, FalsePos: m.FalsePos, TrueNeg: m.TrueNeg, FalseNeg: m.FalseNeg}
}

// isBetterThreshold checks if current metrics are better than best
func isBetterThreshold(current, best NNMetrics) bool {
	if current.F2 > best.F2+1e-12 {
		return true
	}
	if math.Abs(current.F2-best.F2) <= 1e-12 && current.Recall > best.Recall+1e-12 {
		return true
	}
	return false
}

// evaluateNN computes all metrics on a dataset
func evaluateNN(model *FraudNeuralNetwork, inputs, labels *mat.Dense, threshold float64) (NNMetrics, error) {
	predictions, err := model.predict(inputs)
	if err != nil {
		return NNMetrics{}, err
	}
	m, err := irisnn.EvaluateBinaryOutputs(predictions, labels, threshold)
	return nnMetrics(m), err
}

// predict generates predictions from input matrix
func (nn *FraudNeuralNetwork) predict(inputs *mat.Dense) (*mat.Dense, error) {
	if nn.WeightsHid == nil || nn.WeightsOut == nil {
		return nil, fmt.Errorf("network not trained")
	}
	_, predictions := nn.forward(inputs)
	return predictions, nil
}

// predictTransactionNN predicts fraud for a single transaction
func predictTransactionNN(model *FraudNeuralNetwork, tx FraudTransaction) (float64, int, error) {
	features := fraudNumericFeatures(tx)
	normalized := make([]float64, len(features))
	for i := range features {
		normalized[i] = (features[i] - model.Scaler.Mean[i]) / model.Scaler.Std[i]
	}

	featureVector := make([]float64, numInputFeatures)
	copy(featureVector, normalized)

	// One-hot encode transaction type
	typeOffset := len(normalized)
	if idx, ok := model.TypeIndex[strings.ToUpper(tx.Type)]; ok {
		featureVector[typeOffset+idx] = 1
	}

	// Merchant indicator
	if strings.HasPrefix(tx.NameDest, "M") {
		featureVector[len(featureVector)-1] = 1
	}

	// Create single-row matrix
	inputMatrix := mat.NewDense(1, numInputFeatures, featureVector)
	_, predictions := model.forward(inputMatrix)

	probabilities, err := irisnn.BinaryPositiveClassProbabilities(predictions)
	if err != nil {
		return 0, 0, err
	}
	fraudProb := probabilities[0]
	label := 0
	if fraudProb >= model.Threshold {
		label = 1
	}

	return fraudProb, label, nil
}

// predictCSVNN batch predicts on CSV file with worker goroutines
func predictCSVNN(model *FraudNeuralNetwork, inputPath, outputPath string, maxRows, workers int) error {
	inputFile, err := os.Open(inputPath)
	if err != nil {
		return fmt.Errorf("open input: %w", err)
	}
	defer inputFile.Close()

	reader := csv.NewReader(inputFile)
	header, err := reader.Read()
	if err != nil {
		return fmt.Errorf("read header: %w", err)
	}

	outputFile, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("create output: %w", err)
	}
	defer outputFile.Close()

	writer := csv.NewWriter(outputFile)
	defer writer.Flush()

	outHeader := append(append([]string{}, header...), "predicted_probability", "predicted_label")
	if err := writer.Write(outHeader); err != nil {
		return err
	}

	jobs := make(chan struct {
		idx int
		row []string
	}, workers*2)
	results := make(chan struct {
		idx    int
		outRow []string
		err    error
	}, workers*2)

	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for job := range jobs {
				tx, err := parseTransactionRecord(job.row)
				if err != nil {
					results <- struct {
						idx    int
						outRow []string
						err    error
					}{job.idx, nil, err}
					continue
				}
				prob, label, err := predictTransactionNN(model, tx)
				outRow := append(append([]string{}, job.row...), fmt.Sprintf("%.6f", prob), strconv.Itoa(label))
				results <- struct {
					idx    int
					outRow []string
					err    error
				}{job.idx, outRow, err}
			}
		}()
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	readDone := make(chan error, 1)
	go func() {
		defer close(jobs)
		rowIdx := 0
		for {
			row, err := reader.Read()
			if err != nil {
				if err == io.EOF {
					readDone <- nil
					return
				}
				readDone <- fmt.Errorf("read row: %w", err)
				return
			}
			if maxRows > 0 && rowIdx >= maxRows {
				readDone <- nil
				return
			}
			jobs <- struct {
				idx int
				row []string
			}{rowIdx, row}
			rowIdx++
		}
	}()

	pending := make(map[int]struct {
		outRow []string
		err    error
	})
	nextIdx := 0
	for result := range results {
		if result.err != nil {
			return result.err
		}
		pending[result.idx] = struct {
			outRow []string
			err    error
		}{result.outRow, result.err}

		for {
			res, ok := pending[nextIdx]
			if !ok {
				break
			}
			delete(pending, nextIdx)
			if err := writer.Write(res.outRow); err != nil {
				return err
			}
			nextIdx++
		}
	}

	if err := <-readDone; err != nil {
		return err
	}

	return writer.Error()
}

// parseTransactionRecord parses a CSV row into a transaction
func parseTransactionRecord(row []string) (FraudTransaction, error) {
	if len(row) != expectedFieldsNN {
		return FraudTransaction{}, fmt.Errorf("expected %d fields, got %d", expectedFieldsNN, len(row))
	}

	step, err := strconv.ParseFloat(row[0], 64)
	if err != nil {
		return FraudTransaction{}, fmt.Errorf("parse step: %w", err)
	}
	amount, err := strconv.ParseFloat(row[2], 64)
	if err != nil {
		return FraudTransaction{}, fmt.Errorf("parse amount: %w", err)
	}
	oldBalOrig, err := strconv.ParseFloat(row[4], 64)
	if err != nil {
		return FraudTransaction{}, fmt.Errorf("parse oldbalanceOrig: %w", err)
	}
	newBalOrig, err := strconv.ParseFloat(row[5], 64)
	if err != nil {
		return FraudTransaction{}, fmt.Errorf("parse newbalanceOrig: %w", err)
	}
	oldBalDest, err := strconv.ParseFloat(row[7], 64)
	if err != nil {
		return FraudTransaction{}, fmt.Errorf("parse oldbalanceDest: %w", err)
	}
	newBalDest, err := strconv.ParseFloat(row[8], 64)
	if err != nil {
		return FraudTransaction{}, fmt.Errorf("parse newbalanceDest: %w", err)
	}
	isFraud, err := strconv.Atoi(row[9])
	if err != nil {
		return FraudTransaction{}, fmt.Errorf("parse isFraud: %w", err)
	}

	return FraudTransaction{
		Step:           step,
		Type:           row[1],
		Amount:         amount,
		OldBalanceOrig: oldBalOrig,
		NewBalanceOrig: newBalOrig,
		NameDest:       row[6],
		OldBalanceDest: oldBalDest,
		NewBalanceDest: newBalDest,
		IsFraud:        isFraud,
	}, nil
}

// fraudNumericFeatures extracts numeric features from transaction
func fraudNumericFeatures(tx FraudTransaction) []float64 {
	origDelta := tx.OldBalanceOrig - tx.NewBalanceOrig
	destDelta := tx.NewBalanceDest - tx.OldBalanceDest
	return []float64{
		tx.Step,
		tx.Amount,
		tx.OldBalanceOrig,
		tx.NewBalanceOrig,
		tx.OldBalanceDest,
		tx.NewBalanceDest,
		origDelta,
		destDelta,
	}
}

// fraudFeatureNames returns feature column names
func fraudFeatureNames() []string {
	return []string{
		"step_z", "amount_z", "oldbalanceOrig_z", "newbalanceOrig_z",
		"oldbalanceDest_z", "newbalanceDest_z", "origDelta_z", "destDelta_z",
		"type_PAYMENT", "type_TRANSFER", "type_CASH_OUT", "type_DEBIT",
		"type_CASH_IN", "dest_is_merchant",
	}
}

// buildTypeIndex creates map of transaction types to indices
func buildTypeIndex() map[string]int {
	index := make(map[string]int)
	for i, t := range txTypes {
		index[t] = i
	}
	return index
}

// saveFraudModelNN saves model to JSON
func saveFraudModelNN(path string, model *FraudNeuralNetwork) error {
	hidRows, hidCols := model.WeightsHid.Dims()
	outRows, outCols := model.WeightsOut.Dims()

	artifact := FraudModelArtifact{
		Weights: NetworkWeights{
			WeightsHid: mat.DenseCopyOf(model.WeightsHid).RawMatrix().Data,
			BiasHid:    mat.DenseCopyOf(model.BiasHid).RawMatrix().Data,
			WeightsOut: mat.DenseCopyOf(model.WeightsOut).RawMatrix().Data,
			BiasOut:    mat.DenseCopyOf(model.BiasOut).RawMatrix().Data,
			HidShape:   [2]int{hidRows, hidCols},
			OutShape:   [2]int{outRows, outCols},
		},
		Config:       model.Config,
		Scaler:       model.Scaler,
		FeatureNames: model.FeatureNames,
		TypeIndex:    model.TypeIndex,
		Threshold:    model.Threshold,
		Summary:      model.Summary,
	}

	data, err := json.MarshalIndent(artifact, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}

	return os.WriteFile(path, data, 0644)
}

// loadFraudModelNN loads model from JSON
func loadFraudModelNN(path string) (*FraudNeuralNetwork, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read file: %w", err)
	}

	var artifact FraudModelArtifact
	if err := json.Unmarshal(data, &artifact); err != nil {
		return nil, fmt.Errorf("unmarshal: %w", err)
	}

	model := &FraudNeuralNetwork{
		Config:       artifact.Config,
		Scaler:       artifact.Scaler,
		FeatureNames: artifact.FeatureNames,
		TypeIndex:    artifact.TypeIndex,
		Threshold:    artifact.Threshold,
		Summary:      artifact.Summary,
		WeightsHid: mat.NewDense(
			artifact.Weights.HidShape[0],
			artifact.Weights.HidShape[1],
			artifact.Weights.WeightsHid,
		),
		BiasHid: mat.NewDense(1, len(artifact.Weights.BiasHid), artifact.Weights.BiasHid),
		WeightsOut: mat.NewDense(
			artifact.Weights.OutShape[0],
			artifact.Weights.OutShape[1],
			artifact.Weights.WeightsOut,
		),
		BiasOut: mat.NewDense(1, len(artifact.Weights.BiasOut), artifact.Weights.BiasOut),
	}

	return model, nil
}

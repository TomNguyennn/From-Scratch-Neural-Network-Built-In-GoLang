package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
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
	defaultCutoff           = 0.5
	expectedFields          = 10
	numericFeatureCount     = 8
	categoricalFeatureCount = 6
	fraudInputFeatures      = numericFeatureCount + categoricalFeatureCount
	fraudOutputClasses      = 1
	defaultHiddenNeurons    = 16
)

var transactionTypes = []string{"PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"}

type metrics struct {
	Loss      float64 `json:"loss"`
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

type trainingSummary struct {
	Train           metrics       `json:"train"`
	Validation      metrics       `json:"validation"`
	Test            metrics       `json:"test"`
	Epochs          int           `json:"epochs"`
	Elapsed         time.Duration `json:"elapsed"`
	MaxRows         int           `json:"max_rows"`
	LearnedAt       time.Time     `json:"learned_at"`
	LearningRate    float64       `json:"learning_rate"`
	L2              float64       `json:"l2"`
	Threshold       float64       `json:"threshold"`
	ThresholdMetric string        `json:"threshold_metric"`
	ThresholdStep   float64       `json:"threshold_step"`
	TrainInputPath  string        `json:"train_input_path"`
	ValidationPath  string        `json:"validation_input_path"`
	TestPath        string        `json:"test_input_path"`
}

type fraudModel struct {
	Model        irisnn.ModelArtifact  `json:"model"`
	FeatureNames []string              `json:"feature_names"`
	TypeIndex    map[string]int        `json:"type_index"`
	Threshold    float64               `json:"threshold"`
	Summary      trainingSummary       `json:"summary"`
	Network      *irisnn.NeuralNetwork `json:"-"`
	Scaler       *irisnn.FeatureScaler `json:"-"`
}

type fraudRawData struct {
	NumericData     []float64
	CategoricalData []float64
	Labels          []int
	Rows            int
}

type record struct {
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

type predictionJob struct {
	index int
	row   []string
}

type predictionResult struct {
	index     int
	recordOut []string
	err       error
}

func main() {
	mode := flag.String("mode", "train", "train, predict-csv, or predict-one")
	trainInput := flag.String("train-input", "credit-card-fraud-datasets/transactions_train_split.csv", "Path to training CSV")
	validationInput := flag.String("validation-input", "credit-card-fraud-datasets/transactions_validation.csv", "Path to validation CSV")
	testInput := flag.String("test-input", "credit-card-fraud-datasets/transactions_test.csv", "Path to test CSV")
	inputPath := flag.String("input", "credit-card-fraud-datasets/transactions_train.csv", "Path to prediction input CSV")
	modelPath := flag.String("model", "credit-card-fraud-datasets/transactions_model.json", "Path to save or load the fraud model")
	outputPath := flag.String("output", "credit-card-fraud-datasets/transactions_predictions.csv", "Path to write batch predictions")
	epochs := flag.Int("epochs", 2, "Number of training epochs")
	learningRate := flag.Float64("learning-rate", 0.01, "Learning rate for the neural network")
	l2 := flag.Float64("l2", 0.00001, "Compatibility flag kept for the fraud CLI; not applied by irisnn training")
	hiddenNeurons := flag.Int("hidden-neurons", defaultHiddenNeurons, "Number of hidden neurons in the fraud neural network")
	seed := flag.Int("seed", 42, "Random seed for fraud model initialization")
	maxRows := flag.Int("max-rows", 0, "Optional maximum number of rows to read from each CSV; 0 uses all rows")
	thresholdStep := flag.Float64("threshold-step", 0.01, "Threshold search step for validation tuning")
	workers := flag.Int("workers", runtime.NumCPU(), "Number of worker goroutines for batch prediction")

	step := flag.Float64("step", 0, "Transaction step")
	txType := flag.String("type", "", "Transaction type")
	amount := flag.Float64("amount", 0, "Transaction amount")
	oldBalanceOrig := flag.Float64("oldbalance-orig", 0, "Origin account balance before")
	newBalanceOrig := flag.Float64("newbalance-orig", 0, "Origin account balance after")
	nameDest := flag.String("name-dest", "", "Destination account id")
	oldBalanceDest := flag.Float64("oldbalance-dest", 0, "Destination account balance before")
	newBalanceDest := flag.Float64("newbalance-dest", 0, "Destination account balance after")
	flag.Parse()

	switch *mode {
	case "train":
		fmt.Printf("Training with threshold = %.2f\n", defaultCutoff)
		model, err := trainFraudModel(
			*trainInput,
			*validationInput,
			*testInput,
			*epochs,
			*learningRate,
			*l2,
			*hiddenNeurons,
			*seed,
			*maxRows,
			*thresholdStep,
		)
		if err != nil {
			log.Fatal(err)
		}
		if err := saveFraudModel(*modelPath, model); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("Saved model to %s\n", *modelPath)
	case "predict-csv":
		timeStart := time.Now()
		model, err := loadFraudModel(*modelPath)
		if err != nil {
			log.Fatal(err)
		}
		if *workers < 1 {
			log.Fatal("workers must be at least 1")
		}
		if err := predictCSV(model, *inputPath, *outputPath, *maxRows, *workers); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("Batch prediction completed in %s\n", time.Since(timeStart))
		fmt.Printf("Wrote predictions to %s\n", *outputPath)
		fmt.Printf("Processed batch predictions with %d worker goroutines\n", *workers)
	case "predict-one":
		model, err := loadFraudModel(*modelPath)
		if err != nil {
			log.Fatal(err)
		}
		one := record{
			Step:           *step,
			Type:           *txType,
			Amount:         *amount,
			OldBalanceOrig: *oldBalanceOrig,
			NewBalanceOrig: *newBalanceOrig,
			NameDest:       *nameDest,
			OldBalanceDest: *oldBalanceDest,
			NewBalanceDest: *newBalanceDest,
		}
		probability, label, err := predictRecord(model, one)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("Fraud probability: %.6f\n", probability)
		fmt.Printf("Predicted label: %d\n", label)
		fmt.Printf("Threshold used: %.2f\n", model.Threshold)
	default:
		log.Fatalf("unsupported mode %q", *mode)
	}
}

func trainFraudModel(
	trainPath,
	validationPath,
	testPath string,
	epochs int,
	learningRate,
	l2 float64,
	hiddenNeurons,
	seed,
	maxRows int,
	thresholdStep float64,
) (*fraudModel, error) {
	start := time.Now()
	typeIndex := buildTypeIndex()

	trainRaw, err := loadFraudRawData(trainPath, maxRows, typeIndex)
	if err != nil {
		return nil, err
	}
	validationRaw, err := loadFraudRawData(validationPath, maxRows, typeIndex)
	if err != nil {
		return nil, err
	}
	testRaw, err := loadFraudRawData(testPath, maxRows, typeIndex)
	if err != nil {
		return nil, err
	}

	scaler := irisnn.FitFeatureScaler(mat.NewDense(trainRaw.Rows, numericFeatureCount, trainRaw.NumericData))

	trainSet, err := buildFraudDataset(trainRaw, scaler)
	if err != nil {
		return nil, err
	}
	validationSet, err := buildFraudDataset(validationRaw, scaler)
	if err != nil {
		return nil, err
	}
	testSet, err := buildFraudDataset(testRaw, scaler)
	if err != nil {
		return nil, err
	}

	config := irisnn.NeuralNetConfig{
		Loss:          "bce",
		InputNeurons:  fraudInputFeatures,
		OutputNeurons: fraudOutputClasses,
		HiddenNeurons: hiddenNeurons,
		NumEpochs:     epochs,
		LearningRate:  learningRate,
		Seed:          seed,
	}
	network := irisnn.NewNeuralNetwork(config, scaler)
	if err := network.Train(trainSet.Inputs, trainSet.Labels); err != nil {
		return nil, err
	}

	threshold, valMetrics, err := tuneThreshold(network, validationSet, thresholdStep)
	if err != nil {
		return nil, err
	}

	trainMetrics, err := evaluateModel(network, trainSet, threshold)
	if err != nil {
		return nil, err
	}
	testMetrics, err := evaluateModel(network, testSet, threshold)
	if err != nil {
		return nil, err
	}

	fmt.Printf(
		"Training complete - train bce %.6f acc %.4f precision %.4f recall %.4f f2 %.4f | val bce %.6f acc %.4f precision %.4f recall %.4f f2 %.4f | test bce %.6f acc %.4f precision %.4f recall %.4f f2 %.4f | threshold %.2f\n",
		trainMetrics.Loss,
		trainMetrics.Accuracy,
		trainMetrics.Precision,
		trainMetrics.Recall,
		trainMetrics.F2,
		valMetrics.Loss,
		valMetrics.Accuracy,
		valMetrics.Precision,
		valMetrics.Recall,
		valMetrics.F2,
		testMetrics.Loss,
		testMetrics.Accuracy,
		testMetrics.Precision,
		testMetrics.Recall,
		testMetrics.F2,
		threshold,
	)

	summary := trainingSummary{
		Train:           trainMetrics,
		Validation:      valMetrics,
		Test:            testMetrics,
		Epochs:          epochs,
		Elapsed:         time.Since(start),
		MaxRows:         maxRows,
		LearnedAt:       time.Now(),
		LearningRate:    learningRate,
		L2:              l2,
		Threshold:       threshold,
		ThresholdMetric: "F2",
		ThresholdStep:   thresholdStep,
		TrainInputPath:  trainPath,
		ValidationPath:  validationPath,
		TestPath:        testPath,
	}

	model := &fraudModel{
		Model: network.ModelArtifact(
			toEvaluationMetrics(trainMetrics, valMetrics, testMetrics, time.Since(start)),
			fraudLabelNames(),
		),
		FeatureNames: featureNames(),
		TypeIndex:    typeIndex,
		Threshold:    threshold,
		Summary:      summary,
		Network:      network,
		Scaler:       scaler,
	}

	return model, nil
}

func loadFraudRawData(path string, maxRows int, typeIndex map[string]int) (*fraudRawData, error) {
	raw := &fraudRawData{
		NumericData:     make([]float64, 0, 1024*numericFeatureCount),
		CategoricalData: make([]float64, 0, 1024*categoricalFeatureCount),
		Labels:          make([]int, 0, 1024),
	}

	err := streamRecords(path, maxRows, func(idx int, row []string) error {
		rec, err := parseRecord(row)
		if err != nil {
			return fmt.Errorf("row %d: %w", idx+2, err)
		}

		raw.NumericData = append(raw.NumericData, numericFeatures(rec)...)
		raw.CategoricalData = append(raw.CategoricalData, categoricalFeatures(rec, typeIndex)...)
		raw.Labels = append(raw.Labels, rec.IsFraud)
		raw.Rows++
		return nil
	})
	if err != nil {
		return nil, err
	}
	if raw.Rows == 0 {
		return nil, fmt.Errorf("%s has no rows for training or evaluation", path)
	}

	return raw, nil
}

func buildFraudDataset(raw *fraudRawData, scaler *irisnn.FeatureScaler) (*irisnn.Dataset, error) {
	if raw == nil {
		return nil, fmt.Errorf("fraud raw dataset is nil")
	}
	if scaler == nil {
		return nil, fmt.Errorf("feature scaler is nil")
	}

	numericMatrix := mat.NewDense(raw.Rows, numericFeatureCount, raw.NumericData)
	scaledNumeric := scaler.Transform(numericMatrix)
	scaledData := scaledNumeric.RawMatrix().Data

	inputsData := make([]float64, raw.Rows*fraudInputFeatures)
	for row := 0; row < raw.Rows; row++ {
		inputOffset := row * fraudInputFeatures
		numericOffset := row * numericFeatureCount
		categoricalOffset := row * categoricalFeatureCount

		copy(
			inputsData[inputOffset:inputOffset+numericFeatureCount],
			scaledData[numericOffset:numericOffset+numericFeatureCount],
		)
		copy(
			inputsData[inputOffset+numericFeatureCount:inputOffset+fraudInputFeatures],
			raw.CategoricalData[categoricalOffset:categoricalOffset+categoricalFeatureCount],
		)
	}

	labels := make([]float64, raw.Rows)
	for i, y := range raw.Labels {
		if y != 0 && y != 1 {
			return nil, fmt.Errorf("invalid binary label at row %d", i)
		}
		labels[i] = float64(y)
	}
	return irisnn.NewDataset(mat.NewDense(raw.Rows, fraudInputFeatures, inputsData), mat.NewDense(raw.Rows, 1, labels))
}

func tuneThreshold(network *irisnn.NeuralNetwork, validationSet *irisnn.Dataset, step float64) (float64, metrics, error) {
	if step <= 0 || step >= 1 {
		step = 0.01
	}

	predictions, err := network.Predict(validationSet.Inputs)
	if err != nil {
		return 0, metrics{}, err
	}

	probabilities, err := irisnn.BinaryPositiveClassProbabilities(predictions)
	if err != nil {
		return 0, metrics{}, err
	}
	actuals, err := irisnn.BinaryActualLabels(validationSet.Labels)
	if err != nil {
		return 0, metrics{}, err
	}
	if len(probabilities) == 0 {
		return defaultCutoff, metrics{}, fmt.Errorf("validation set has no rows for threshold tuning")
	}

	bestThreshold := defaultCutoff
	bestMetrics := metrics{F2: -1}

	for threshold := step; threshold < 1.0; threshold += step {
		binaryMetrics, err := irisnn.EvaluateBinaryThreshold(probabilities, actuals, threshold)
		if err != nil {
			return 0, metrics{}, err
		}
		current := toMetrics(binaryMetrics)
		if betterThresholdCandidate(current, threshold, bestMetrics, bestThreshold) {
			bestThreshold = threshold
			bestMetrics = current
		}
	}

	if bestMetrics.F2 < 0 {
		binaryMetrics, err := irisnn.EvaluateBinaryThreshold(probabilities, actuals, defaultCutoff)
		if err != nil {
			return 0, metrics{}, err
		}
		bestMetrics = toMetrics(binaryMetrics)
		bestThreshold = defaultCutoff
	}

	return bestThreshold, bestMetrics, nil
}

func betterThresholdCandidate(current metrics, currentThreshold float64, best metrics, bestThreshold float64) bool {
	if current.F2 > best.F2+1e-12 {
		return true
	}
	if math.Abs(current.F2-best.F2) <= 1e-12 {
		if current.Recall > best.Recall+1e-12 {
			return true
		}
		if math.Abs(current.Recall-best.Recall) <= 1e-12 {
			if current.Precision > best.Precision+1e-12 {
				return true
			}
			if math.Abs(current.Precision-best.Precision) <= 1e-12 && currentThreshold < bestThreshold {
				return true
			}
		}
	}
	return false
}

func evaluateModel(network *irisnn.NeuralNetwork, dataset *irisnn.Dataset, threshold float64) (metrics, error) {
	predictions, err := network.Predict(dataset.Inputs)
	if err != nil {
		return metrics{}, err
	}

	binaryMetrics, err := irisnn.EvaluateBinaryOutputs(predictions, dataset.Labels, threshold)
	if err != nil {
		return metrics{}, err
	}

	return toMetrics(binaryMetrics), nil
}

func predictCSV(model *fraudModel, inputPath, outputPath string, maxRows int, workers int) error {
	inputFile, err := os.Open(inputPath)
	if err != nil {
		return fmt.Errorf("open input csv: %w", err)
	}
	defer inputFile.Close()

	reader := csv.NewReader(inputFile)
	header, err := reader.Read()
	if err != nil {
		return fmt.Errorf("read header: %w", err)
	}

	outputFile, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("create output csv: %w", err)
	}
	defer outputFile.Close()

	writer := csv.NewWriter(outputFile)
	defer writer.Flush()

	outHeader := append([]string{}, header...)
	outHeader = append(outHeader, "predicted_probability", "predicted_label")
	if err := writer.Write(outHeader); err != nil {
		return err
	}

	jobs := make(chan predictionJob, workers*2)
	results := make(chan predictionResult, workers*2)

	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go predictionWorker(model, jobs, results, &wg)
	}
	go func() {
		wg.Wait()
		close(results)
	}()

	readErrCh := make(chan error, 1)
	go func() {
		defer close(jobs)
		rowIndex := 0
		for {
			row, err := reader.Read()
			if err != nil {
				if err == io.EOF {
					readErrCh <- nil
					return
				}
				readErrCh <- fmt.Errorf("read row: %w", err)
				return
			}
			if maxRows > 0 && rowIndex >= maxRows {
				readErrCh <- nil
				return
			}
			jobs <- predictionJob{
				index: rowIndex,
				row:   row,
			}
			rowIndex++
		}
	}()

	pending := make(map[int]predictionResult)
	nextIndex := 0
	for result := range results {
		if result.err != nil {
			return result.err
		}
		pending[result.index] = result

		for {
			current, ok := pending[nextIndex]
			if !ok {
				break
			}
			delete(pending, nextIndex)
			if err := writer.Write(current.recordOut); err != nil {
				return err
			}
			nextIndex++
		}
	}

	if err := <-readErrCh; err != nil {
		return err
	}

	if err := writer.Error(); err != nil {
		return err
	}
	return nil
}

func predictionWorker(model *fraudModel, jobs <-chan predictionJob, results chan<- predictionResult, wg *sync.WaitGroup) {
	defer wg.Done()
	for job := range jobs {
		rec, err := parseRecord(job.row)
		if err != nil {
			results <- predictionResult{
				index: job.index,
				err:   fmt.Errorf("row %d: %w", job.index+2, err),
			}
			continue
		}
		probability, label, err := predictRecord(model, rec)
		if err != nil {
			results <- predictionResult{
				index: job.index,
				err:   fmt.Errorf("row %d: %w", job.index+2, err),
			}
			continue
		}
		recordOut := append([]string{}, job.row...)
		recordOut = append(recordOut, fmt.Sprintf("%.6f", probability), strconv.Itoa(label))
		results <- predictionResult{
			index:     job.index,
			recordOut: recordOut,
		}
	}
}

func predictRecord(model *fraudModel, rec record) (float64, int, error) {
	inputMatrix, err := buildPredictionInput(model, rec)
	if err != nil {
		return 0, 0, err
	}

	predictions, err := model.Network.Predict(inputMatrix)
	if err != nil {
		return 0, 0, err
	}

	probabilities, err := irisnn.BinaryPositiveClassProbabilities(predictions)
	if err != nil {
		return 0, 0, err
	}
	probability := probabilities[0]
	label := 0
	if probability >= model.Threshold {
		label = 1
	}
	return probability, label, nil
}

func buildPredictionInput(model *fraudModel, rec record) (*mat.Dense, error) {
	if model == nil || model.Network == nil {
		return nil, fmt.Errorf("fraud model is not initialized")
	}
	if model.Scaler == nil {
		return nil, fmt.Errorf("fraud model scaler is not initialized")
	}

	numeric := numericFeatures(rec)
	scaledNumeric := model.Scaler.Transform(mat.NewDense(1, numericFeatureCount, numeric))
	inputsData := append([]float64{}, scaledNumeric.RawMatrix().Data...)
	inputsData = append(inputsData, categoricalFeatures(rec, model.TypeIndex)...)

	return mat.NewDense(1, fraudInputFeatures, inputsData), nil
}

func parseRecord(row []string) (record, error) {
	if len(row) != expectedFields {
		return record{}, fmt.Errorf("expected %d columns, got %d", expectedFields, len(row))
	}

	step, err := strconv.ParseFloat(row[0], 64)
	if err != nil {
		return record{}, fmt.Errorf("parse step: %w", err)
	}
	amount, err := strconv.ParseFloat(row[2], 64)
	if err != nil {
		return record{}, fmt.Errorf("parse amount: %w", err)
	}
	oldBalanceOrig, err := strconv.ParseFloat(row[4], 64)
	if err != nil {
		return record{}, fmt.Errorf("parse oldbalanceOrig: %w", err)
	}
	newBalanceOrig, err := strconv.ParseFloat(row[5], 64)
	if err != nil {
		return record{}, fmt.Errorf("parse newbalanceOrig: %w", err)
	}
	oldBalanceDest, err := strconv.ParseFloat(row[7], 64)
	if err != nil {
		return record{}, fmt.Errorf("parse oldbalanceDest: %w", err)
	}
	newBalanceDest, err := strconv.ParseFloat(row[8], 64)
	if err != nil {
		return record{}, fmt.Errorf("parse newbalanceDest: %w", err)
	}
	isFraud, err := strconv.Atoi(row[9])
	if err != nil {
		return record{}, fmt.Errorf("parse isFraud: %w", err)
	}

	return record{
		Step:           step,
		Type:           row[1],
		Amount:         amount,
		OldBalanceOrig: oldBalanceOrig,
		NewBalanceOrig: newBalanceOrig,
		NameDest:       row[6],
		OldBalanceDest: oldBalanceDest,
		NewBalanceDest: newBalanceDest,
		IsFraud:        isFraud,
	}, nil
}

func numericFeatures(rec record) []float64 {
	origDelta := rec.OldBalanceOrig - rec.NewBalanceOrig
	destDelta := rec.NewBalanceDest - rec.OldBalanceDest
	return []float64{
		rec.Step,
		rec.Amount,
		rec.OldBalanceOrig,
		rec.NewBalanceOrig,
		rec.OldBalanceDest,
		rec.NewBalanceDest,
		origDelta,
		destDelta,
	}
}

func categoricalFeatures(rec record, typeIndex map[string]int) []float64 {
	features := make([]float64, categoricalFeatureCount)

	if idx, ok := typeIndex[strings.ToUpper(rec.Type)]; ok {
		features[idx] = 1
	}
	if strings.HasPrefix(rec.NameDest, "M") {
		features[len(features)-1] = 1
	}

	return features
}

func featureNames() []string {
	return []string{
		"step_z",
		"amount_z",
		"oldbalanceOrig_z",
		"newbalanceOrig_z",
		"oldbalanceDest_z",
		"newbalanceDest_z",
		"origDelta_z",
		"destDelta_z",
		"type_PAYMENT",
		"type_TRANSFER",
		"type_CASH_OUT",
		"type_DEBIT",
		"type_CASH_IN",
		"dest_is_merchant",
	}
}

func buildTypeIndex() map[string]int {
	index := make(map[string]int, len(transactionTypes))
	for i, name := range transactionTypes {
		index[name] = i
	}
	return index
}

func fraudLabelNames() map[int]string {
	return map[int]string{
		0: "not_fraud",
		1: "fraud",
	}
}

func streamRecords(path string, maxRows int, fn func(idx int, row []string) error) error {
	file, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open csv: %w", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	if _, err := reader.Read(); err != nil {
		return fmt.Errorf("read header: %w", err)
	}

	rowIndex := 0
	for {
		row, err := reader.Read()
		if err != nil {
			if err == io.EOF {
				break
			}
			return fmt.Errorf("read row: %w", err)
		}
		if maxRows > 0 && rowIndex >= maxRows {
			break
		}
		if err := fn(rowIndex, row); err != nil {
			return err
		}
		rowIndex++
	}
	return nil
}

func saveFraudModel(path string, model *fraudModel) error {
	if model == nil {
		return fmt.Errorf("fraud model is nil")
	}
	if model.Network != nil {
		model.Model = model.Network.ModelArtifact(
			toEvaluationMetrics(model.Summary.Train, model.Summary.Validation, model.Summary.Test, model.Summary.Elapsed),
			fraudLabelNames(),
		)
	}

	data, err := json.MarshalIndent(model, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

func loadFraudModel(path string) (*fraudModel, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var model fraudModel
	if err := json.Unmarshal(data, &model); err != nil {
		return nil, err
	}

	network, scaler, err := irisnn.LoadModelArtifact(&model.Model)
	if err != nil {
		return nil, err
	}

	model.Network = network
	model.Scaler = scaler

	return &model, nil
}

func toMetrics(binary irisnn.BinaryMetrics) metrics {
	return metrics{
		Loss:      binary.Loss,
		Accuracy:  binary.Accuracy,
		Precision: binary.Precision,
		Recall:    binary.Recall,
		F1:        binary.F1,
		F2:        binary.F2,
		PosRate:   binary.PosRate,
		Rows:      binary.Rows,
		TruePos:   binary.TruePos,
		FalsePos:  binary.FalsePos,
		TrueNeg:   binary.TrueNeg,
		FalseNeg:  binary.FalseNeg,
	}
}

func toEvaluationMetrics(train, validation, test metrics, duration time.Duration) irisnn.EvaluationMetrics {
	return irisnn.EvaluationMetrics{
		TrainLoss: train.Loss,
		ValLoss:   validation.Loss,
		TestLoss:  test.Loss,
		TrainAcc:  train.Accuracy,
		ValAcc:    validation.Accuracy,
		TestAcc:   test.Accuracy,
		Duration:  duration,
		Timestamp: time.Now(),
	}
}

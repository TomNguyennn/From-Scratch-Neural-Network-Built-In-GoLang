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
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"go_neural_network/internal/irisnn"

	"gonum.org/v1/gonum/mat"
)

const (
	defaultCutoff           = 0.3
	expectedFields          = 10
	numericFeatureCount     = 1
	categoricalFeatureCount = 6
	fraudInputFeatures      = 11
	fraudOutputClasses      = 1
	defaultHiddenNeurons    = 16
)

var transactionTypes = []string{"PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"}

type metrics struct {
	ProbabilityStats *irisnn.ClassProbabilityStats `json:"probability_stats,omitempty"`
	WeightedBCE      float64                       `json:"weighted_bce"`
	AveragePrecision float64                       `json:"average_precision"`
	Loss             float64                       `json:"loss"`
	Accuracy         float64                       `json:"accuracy"`
	Precision        float64                       `json:"precision"`
	Recall           float64                       `json:"recall"`
	F1               float64                       `json:"f1"`
	F2               float64                       `json:"f2"`
	PosRate          float64                       `json:"positive_rate"`
	Rows             int64                         `json:"rows"`
	TruePos          int64                         `json:"true_pos"`
	FalsePos         int64                         `json:"false_pos"`
	TrueNeg          int64                         `json:"true_neg"`
	FalseNeg         int64                         `json:"false_neg"`
}

type thresholdResult struct {
	Threshold float64 `json:"threshold"`
	Precision float64 `json:"precision"`
	Recall    float64 `json:"recall"`
	F2        float64 `json:"f2"`
}

type trainingSummary struct {
	ValidationThresholds []thresholdResult `json:"validation_thresholds"`
	Train                metrics           `json:"train"`
	Validation           metrics           `json:"validation"`
	Test                 metrics           `json:"test"`
	Epochs               int               `json:"epochs"`
	Elapsed              time.Duration     `json:"elapsed"`
	MaxRows              int               `json:"max_rows"`
	LearnedAt            time.Time         `json:"learned_at"`
	LearningRate         float64           `json:"learning_rate"`
	L2                   float64           `json:"l2"`
	Threshold            float64           `json:"threshold"`
	ThresholdMetric      string            `json:"threshold_metric"`
	TrainInputPath       string            `json:"train_input_path"`
	ValidationPath       string            `json:"validation_input_path"`
	TestPath             string            `json:"test_input_path"`
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
	NumericData []float64
	Records     []record
	Labels      []int
	Rows        int
}

type record struct {
	Step     float64
	Type     string
	Amount   float64
	NameDest string
	IsFraud  int
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
	modelPath := flag.String("model", "new_transaction_features_2_layer_recommend_settings.json", "Path to save or load the fraud model")
	outputPath := flag.String("output", "credit-card-fraud-datasets/transactions_predictions.csv", "Path to write batch predictions")
	optimizer := flag.String("optimizer", "adam", "Optimizer: adam or sgd")
	batchSize := flag.Int("batch-size", 256, "Training batch size (positive)")
	epochs := flag.Int("epochs", 50, "Number of training epochs")
	learningRate := flag.Float64("learning-rate", 0.001, "Learning rate for the neural network")
	l2 := flag.Float64("l2", 0.00001, "Compatibility flag kept for the fraud CLI; not applied by irisnn training")
	hiddenNeurons2 := flag.Int("hidden-neurons-2", 32, "Number of neurons in the second ReLU hidden layer")
	hiddenNeurons := flag.Int("hidden-neurons", 64, "Number of hidden neurons in the fraud neural network")
	seed := flag.Int("seed", 42, "Random seed for fraud model initialization")
	maxRows := flag.Int("max-rows", 0, "Optional maximum number of rows to read from each CSV; 0 uses all rows")
	positiveWeight := flag.String("positive-weight", "auto", "Positive BCE weight: auto uses training legitimate/fraud ratio, or specify a positive number")
	workers := flag.Int("workers", runtime.NumCPU(), "Number of worker goroutines for batch prediction")

	step := flag.Float64("step", 0, "Transaction step")
	txType := flag.String("type", "", "Transaction type")
	amount := flag.Float64("amount", 0, "Transaction amount")
	nameDest := flag.String("name-dest", "", "Destination account id")
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
			*hiddenNeurons2,
			*seed,
			*maxRows,
			*positiveWeight,
			*batchSize,
			*optimizer,
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
			Step:     *step,
			Type:     *txType,
			Amount:   *amount,
			NameDest: *nameDest,
		}
		probability, label, err := predictRecord(model, one)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("Fraud probability: %.6f\n", probability)
		fmt.Printf("Predicted label: %d\n", label)
		fmt.Printf("Threshold used: %.17g\n", model.Threshold)
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
	hiddenNeurons2,
	seed,
	maxRows int,
	positiveWeightOption string,
	batchSize int,
	optimizer string,
) (*fraudModel, error) {
	if optimizer != "adam" && optimizer != "sgd" {
		return nil, fmt.Errorf("optimizer must be adam or sgd")
	}
	fmt.Printf("Optimizer: %s\n", optimizer)
	if batchSize <= 0 {
		return nil, fmt.Errorf("batch size must be positive")
	}
	if hiddenNeurons <= 0 || hiddenNeurons2 <= 0 {
		return nil, fmt.Errorf("hidden layer widths must be positive")
	}
	start := time.Now()
	typeIndex := buildTypeIndex()

	trainRaw, err := loadFraudRawData(trainPath, maxRows, typeIndex)
	if err != nil {
		return nil, err
	}
	positiveWeight, err := resolvePositiveWeight(trainRaw.Labels, positiveWeightOption)
	if err != nil {
		return nil, err
	}
	fmt.Printf("Positive-class weight: %.6f\n", positiveWeight)
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
		Loss:           "bce",
		BatchSize:      batchSize,
		Optimizer:      optimizer,
		PositiveWeight: positiveWeight,
		InputNeurons:   fraudInputFeatures,
		OutputNeurons:  fraudOutputClasses,
		HiddenNeurons:  hiddenNeurons,
		HiddenNeurons2: hiddenNeurons2,
		NumEpochs:      epochs,
		LearningRate:   learningRate,
		Seed:           seed,
	}
	network := irisnn.NewNeuralNetwork(config, scaler)
	if err := network.TrainWithEpochCallback(trainSet.Inputs, trainSet.Labels, func(epoch int) error {
		train, err := evaluateModel(network, trainSet, defaultCutoff)
		if err != nil {
			return err
		}
		val, err := evaluateModel(network, validationSet, defaultCutoff)
		if err != nil {
			return err
		}
		fmt.Printf("Epoch %d/%d - train bce %.6f weighted_bce %.6f ap %.6f | val bce %.6f weighted_bce %.6f ap %.6f\n", epoch, epochs, train.Loss, train.WeightedBCE, train.AveragePrecision, val.Loss, val.WeightedBCE, val.AveragePrecision)
		return nil
	}); err != nil {
		return nil, err
	}

	threshold, valMetrics, thresholdRows, err := tuneThreshold(network, validationSet)
	if err != nil {
		return nil, err
	}

	fmt.Println("Validation thresholds:")
	fmt.Println("Threshold    Precision    Recall    F2")
	for _, row := range thresholdRows {
		fmt.Printf("%.2f         %.6f     %.6f  %.6f\n", row.Threshold, row.Precision, row.Recall, row.F2)
	}

	trainMetrics, err := evaluateModelWithStats(network, trainSet, threshold, true)
	if err != nil {
		return nil, err
	}
	testMetrics, err := evaluateModelWithStats(network, testSet, threshold, true)
	if err != nil {
		return nil, err
	}

	fmt.Printf(
		"Training complete - train bce %.6f weighted_bce %.6f acc %.4f precision %.4f recall %.4f f2 %.4f ap %.6f | val bce %.6f weighted_bce %.6f acc %.4f precision %.4f recall %.4f f2 %.4f ap %.6f | test bce %.6f weighted_bce %.6f acc %.4f precision %.4f recall %.4f f2 %.4f ap %.6f | threshold %.17g\n",
		trainMetrics.Loss,
		trainMetrics.WeightedBCE,
		trainMetrics.Accuracy,
		trainMetrics.Precision,
		trainMetrics.Recall,
		trainMetrics.F2, trainMetrics.AveragePrecision,
		valMetrics.Loss,
		valMetrics.WeightedBCE,
		valMetrics.Accuracy,
		valMetrics.Precision,
		valMetrics.Recall,
		valMetrics.F2, valMetrics.AveragePrecision,
		testMetrics.Loss,
		testMetrics.WeightedBCE,
		testMetrics.Accuracy,
		testMetrics.Precision,
		testMetrics.Recall,
		testMetrics.F2, testMetrics.AveragePrecision,
		threshold,
	)

	for _, split := range []struct {
		name  string
		stats *irisnn.ClassProbabilityStats
	}{{"Train", trainMetrics.ProbabilityStats}, {"Validation", valMetrics.ProbabilityStats}, {"Test", testMetrics.ProbabilityStats}} {
		fmt.Printf("%s predicted fraud probabilities by true class:\n", split.name)
		for _, class := range []struct {
			name  string
			stats *irisnn.ProbabilityStats
		}{{"Legitimate", split.stats.Legitimate}, {"Fraud", split.stats.Fraud}} {
			if class.stats == nil {
				fmt.Printf("  %s: count=0, statistics N/A\n", class.name)
				continue
			}
			st := class.stats
			fmt.Printf("  %s: count=%d mean=%.6f median=%.6f p90=%.6f p99=%.6f\n", class.name, st.Count, st.Mean, st.Median, st.P90, st.P99)
		}
	}
	summary := trainingSummary{
		ValidationThresholds: thresholdRows,
		Train:                trainMetrics,
		Validation:           valMetrics,
		Test:                 testMetrics,
		Epochs:               epochs,
		Elapsed:              time.Since(start),
		MaxRows:              maxRows,
		LearnedAt:            time.Now(),
		LearningRate:         learningRate,
		L2:                   l2,
		Threshold:            threshold,
		ThresholdMetric:      "F2",
		TrainInputPath:       trainPath,
		ValidationPath:       validationPath,
		TestPath:             testPath,
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
		NumericData: make([]float64, 0, 1024*numericFeatureCount),
		Records:     make([]record, 0, 1024),
		Labels:      make([]int, 0, 1024),
	}

	err := streamRecords(path, maxRows, func(idx int, row []string) error {
		rec, err := parseRecord(row)
		if err != nil {
			return fmt.Errorf("row %d: %w", idx+2, err)
		}

		raw.NumericData = append(raw.NumericData, numericFeatures(rec)...)
		raw.Records = append(raw.Records, rec)
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

	inputsData := make([]float64, 0, raw.Rows*fraudInputFeatures)
	for _, rec := range raw.Records {
		features, err := encodeFeatures(rec, scaler)
		if err != nil {
			return nil, err
		}
		inputsData = append(inputsData, features...)
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

func tuneThreshold(network *irisnn.NeuralNetwork, validationSet *irisnn.Dataset) (float64, metrics, []thresholdResult, error) {
	predictions, err := predictInChunks(network, validationSet.Inputs)
	if err != nil {
		return 0, metrics{}, nil, err
	}

	probabilities, err := irisnn.BinaryPositiveClassProbabilities(predictions)
	if err != nil {
		return 0, metrics{}, nil, err
	}
	actuals, err := irisnn.BinaryActualLabels(validationSet.Labels)
	if err != nil {
		return 0, metrics{}, nil, err
	}
	if len(probabilities) == 0 {
		return defaultCutoff, metrics{}, nil, fmt.Errorf("validation set has no rows for threshold tuning")
	}

	rows, err := validationThresholdReport(probabilities, actuals)
	if err != nil {
		return 0, metrics{}, nil, err
	}
	binaryMetrics, err := irisnn.BestF2Threshold(probabilities, actuals)
	if err != nil {
		return 0, metrics{}, nil, fmt.Errorf("validation threshold tuning: %w", err)
	}
	bestThreshold := binaryMetrics.Threshold
	bestMetrics := toMetrics(binaryMetrics)

	bestMetrics.AveragePrecision, err = irisnn.AveragePrecision(probabilities, actuals)
	if err != nil {
		return 0, metrics{}, nil, err
	}
	stats, err := irisnn.SummarizeClassProbabilities(probabilities, actuals)
	if err != nil {
		return 0, metrics{}, nil, err
	}
	bestMetrics.ProbabilityStats = &stats
	bestMetrics.WeightedBCE, err = irisnn.WeightedBinaryCrossEntropy(predictions, validationSet.Labels, network.PositiveWeight())
	return bestThreshold, bestMetrics, rows, err
}

func evaluateModel(network *irisnn.NeuralNetwork, dataset *irisnn.Dataset, threshold float64) (metrics, error) {
	return evaluateModelWithStats(network, dataset, threshold, false)
}

func evaluateModelWithStats(network *irisnn.NeuralNetwork, dataset *irisnn.Dataset, threshold float64, includeStats bool) (metrics, error) {
	predictions, err := predictInChunks(network, dataset.Inputs)
	if err != nil {
		return metrics{}, err
	}

	binaryMetrics, err := irisnn.EvaluateBinaryOutputs(predictions, dataset.Labels, threshold)
	if err != nil {
		return metrics{}, err
	}

	result := toMetrics(binaryMetrics)
	probabilities, err := irisnn.BinaryPositiveClassProbabilities(predictions)
	if err != nil {
		return metrics{}, err
	}
	actuals, err := irisnn.BinaryActualLabels(dataset.Labels)
	if err != nil {
		return metrics{}, err
	}
	result.WeightedBCE, err = irisnn.WeightedBinaryCrossEntropy(predictions, dataset.Labels, network.PositiveWeight())
	if err != nil {
		return metrics{}, err
	}
	if includeStats {
		stats, err := irisnn.SummarizeClassProbabilities(probabilities, actuals)
		if err != nil {
			return metrics{}, err
		}
		result.ProbabilityStats = &stats
	}
	result.AveragePrecision, err = irisnn.AveragePrecision(probabilities, actuals)
	return result, err
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

	inputsData, err := encodeFeatures(rec, model.Scaler)
	if err != nil {
		return nil, err
	}
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
	isFraud, err := strconv.Atoi(row[9])
	if err != nil {
		return record{}, fmt.Errorf("parse isFraud: %w", err)
	}

	rec := record{
		Step:     step,
		Type:     row[1],
		Amount:   amount,
		NameDest: row[6],
		IsFraud:  isFraud,
	}
	return rec, validateRecord(rec)
}

func numericFeatures(rec record) []float64 { return []float64{math.Log1p(rec.Amount)} }

func validateRecord(rec record) error {
	if math.IsNaN(rec.Amount) || math.IsInf(rec.Amount, 0) || rec.Amount < 0 {
		return fmt.Errorf("amount must be finite and nonnegative")
	}
	if math.IsNaN(rec.Step) || math.IsInf(rec.Step, 0) || rec.Step < 0 || math.Trunc(rec.Step) != rec.Step {
		return fmt.Errorf("step must be a finite nonnegative integer")
	}
	if _, ok := buildTypeIndex()[strings.ToUpper(rec.Type)]; !ok {
		return fmt.Errorf("unsupported transaction type %q", rec.Type)
	}
	return nil
}

func validAmountScaler(s *irisnn.FeatureScaler) bool {
	return s != nil && len(s.Mean) == 1 && len(s.Std) == 1 && !math.IsNaN(s.Mean[0]) && !math.IsInf(s.Mean[0], 0) && !math.IsNaN(s.Std[0]) && !math.IsInf(s.Std[0], 0) && s.Std[0] > 0
}

func encodeFeatures(rec record, scaler *irisnn.FeatureScaler) ([]float64, error) {
	if err := validateRecord(rec); err != nil {
		return nil, err
	}
	if !validAmountScaler(scaler) {
		return nil, fmt.Errorf("invalid log-amount scaler; retrain the model")
	}
	z := (math.Log1p(rec.Amount) - scaler.Mean[0]) / scaler.Std[0]
	angle := 2 * math.Pi * math.Mod(rec.Step, 24) / 24
	features := []float64{z, 0, 0, math.Sin(angle), math.Cos(angle)}
	switch strings.ToUpper(rec.Type) {
	case "TRANSFER":
		features[1] = z
	case "CASH_OUT":
		features[2] = z
	}
	return append(features, categoricalFeatures(rec, buildTypeIndex())...), nil
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
		"log_amount_z", "transfer_amount", "cashout_amount", "hour_sin", "hour_cos",
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

	if !reflect.DeepEqual(model.FeatureNames, featureNames()) || model.Model.InputWidth() != fraudInputFeatures || !validAmountScaler(model.Model.ScalerStats) {
		return nil, fmt.Errorf("incompatible feature schema; retrain with the balance-free feature pipeline")
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

func resolvePositiveWeight(labels []int, option string) (float64, error) {
	if option != "auto" {
		weight, err := strconv.ParseFloat(option, 64)
		if err != nil || weight <= 0 || math.IsNaN(weight) || math.IsInf(weight, 0) {
			return 0, fmt.Errorf("positive weight must be auto or a finite number greater than zero")
		}
		return weight, nil
	}
	frauds := 0
	for _, label := range labels {
		if label == 1 {
			frauds++
		} else if label != 0 {
			return 0, fmt.Errorf("invalid training label %d", label)
		}
	}
	legitimate := len(labels) - frauds
	if frauds == 0 || legitimate == 0 {
		return 0, fmt.Errorf("automatic positive weight requires both classes in the loaded training rows")
	}
	return float64(legitimate) / float64(frauds), nil
}

func validationThresholdReport(probabilities []float64, actuals []int) ([]thresholdResult, error) {
	var rows []thresholdResult
	for _, threshold := range []float64{0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80} {
		m, err := irisnn.EvaluateBinaryThreshold(probabilities, actuals, threshold)
		if err != nil {
			return nil, err
		}
		rows = append(rows, thresholdResult{threshold, m.Precision, m.Recall, m.F2})
	}
	return rows, nil
}

// Bound hidden activation memory while retaining probabilities for exact AP.
func predictInChunks(network *irisnn.NeuralNetwork, inputs *mat.Dense) (*mat.Dense, error) {
	rows, cols := inputs.Dims()
	result := mat.NewDense(rows, 1, nil)
	for start := 0; start < rows; start += 512 {
		end := min(start+512, rows)
		prediction, err := network.Predict(inputs.Slice(start, end, 0, cols).(*mat.Dense))
		if err != nil {
			return nil, err
		}
		for row := start; row < end; row++ {
			result.Set(row, 0, prediction.At(row-start, 0))
		}
	}
	return result, nil
}

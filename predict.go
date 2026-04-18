package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"go_neural_network/internal/irisnn"
	"log"
	"os"
	"strconv"
	"sync"
	"time"

	"gonum.org/v1/gonum/mat"
)

type Job struct {
	ID        int
	InputData []string
}
type Result struct {
	Index         int
	InputData     []string
	Prediction    string
	Probabilities []float64
	Err           error
}

func predictWorker(model *irisnn.NeuralNetwork, scaler *irisnn.FeatureScaler, labelNames map[int]string, jobs <-chan Job, results chan<- Result, wg *sync.WaitGroup) {
	defer wg.Done()
	for job := range jobs {

		features, err := extractFeatures(job.InputData)
		if err != nil {
			results <- Result{Index: job.ID, InputData: job.InputData, Err: err}
			continue
		}

		newMatrix := mat.NewDense(1, 4, features)
		normalisedInput := scaler.Transform(newMatrix)
		output, err := model.Predict(normalisedInput)
		if err != nil {
			results <- Result{Index: job.ID, InputData: job.InputData, Err: err}
			continue
		}

		outputArray := append([]float64(nil), output.RawMatrix().Data...)
		bestIndex := 0
		bestProb := 0.0

		for i, prob := range outputArray {
			if prob > bestProb {
				bestIndex = i
				bestProb = prob
			}
		}

		results <- Result{
			Index:         job.ID,
			InputData:     append([]string(nil), job.InputData...),
			Prediction:    labelNames[bestIndex],
			Probabilities: outputArray,
			Err:           err,
		}
	}
}

func runBatchMode(model *irisnn.NeuralNetwork, scaler *irisnn.FeatureScaler, labelNames map[int]string, inputCSV, outputCSV string, workers, repeats int) error {
	file, err := os.Open(inputCSV)

	if err != nil {
		return fmt.Errorf("open input csv: %w", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	rows, err := reader.ReadAll()
	if err != nil {
		return fmt.Errorf("read input csv: %w", err)

	}

	if len(rows) < 2 {
		return fmt.Errorf("input csv must contain a header and at least one data row")
	}
	if workers < 1 {
		return fmt.Errorf("workers must be at least 1")
	}
	if repeats < 1 {
		return fmt.Errorf("benchmark-repeats must be at least 1")
	}

	header := append([]string{}, rows[0]...)
	header = append(header, "PredictedSpecies", "ProbSetosa", "ProbVersicolor", "ProbVirginica")
	dataRows := rows[1:]
	var finalOutput []Result

	for i := 0; i < repeats; i++ {
		start := time.Now()
		//channels for worker pool
		jobs := make(chan Job)
		results := make(chan Result)

		//workers
		var wg sync.WaitGroup
		for w := 0; w < workers; w++ {
			wg.Add(1)
			go predictWorker(model, scaler, labelNames, jobs, results, &wg)
		}

		//send jobs
		go func() {
			for id, row := range dataRows {
				jobs <- Job{
					ID:        id,
					InputData: row,
				}
			}
			close(jobs)
		}()

		//collect the results
		go func() {
			wg.Wait()
			close(results)
		}()

		localOutput := make([]Result, len(dataRows))
		var firstErr error

		for result := range results {
			if result.Err != nil {
				if firstErr == nil {
					firstErr = fmt.Errorf("row %d: %w", result.Index+2, result.Err)
				}
				continue
			}
			localOutput[result.Index] = result
		}

		fmt.Printf("Benchmark run %d took %v\n", i+1, time.Since(start))

		if firstErr != nil {
			return firstErr
		}
		finalOutput = localOutput
	}

	if err := writePredictionCSV(outputCSV, header, finalOutput); err != nil {
		return err
	}

	fmt.Printf("Wrote batch predictions to %s\n", outputCSV)
	return nil
}

func runSingleMode(scaler *irisnn.FeatureScaler, model *irisnn.NeuralNetwork, artifact *irisnn.ModelArtifact, sl, sw, pl, pw float64) error {
	//input
	input := []float64{sl, sw, pl, pw}
	matrix := mat.NewDense(1, 4, input)

	//normalise

	normalisedInput := scaler.Transform(matrix)

	output, err := model.Predict(normalisedInput)

	if err != nil {
		return fmt.Errorf("Error occured: %e", err)
	}

	outputArray := output.RawMatrix().Data
	bestIndex := 0
	bestProb := 0.0

	for i, prob := range outputArray {
		if prob > bestProb {
			bestIndex = i
			bestProb = prob
		}
	}
	fmt.Printf("Predicted Species: %s\n", artifact.LabelNames[bestIndex])
	fmt.Printf("Class Probabilities: %v\n", outputArray)

	return nil

}

func extractFeatures(row []string) ([]float64, error) {
	var featureFields []string

	//normalise
	switch {
	case len(row) >= 5:
		featureFields = row[1:5]
	case len(row) == 4:
		featureFields = row
	default:
		return nil, fmt.Errorf("expected at least 4 feature columns, got %d", len(row))
	}

	features := make([]float64, 4)
	for i, field := range featureFields {
		value, err := strconv.ParseFloat(field, 64)
		if err != nil {
			return nil, fmt.Errorf("parse feature %d: %w", i, err)
		}
		features[i] = value
	}

	return features, nil
}

func writePredictionCSV(path string, header []string, results []Result) (err error) {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create output csv: %w", err)
	}

	defer func() {
		closeErr := file.Close()
		if err == nil && closeErr != nil {
			err = closeErr
		}
	}()
	writer := csv.NewWriter(file)
	if err := writer.Write(header); err != nil {
		return err
	}
	for _, result := range results {
		record := append([]string{}, result.InputData...)
		record = append(
			record,
			result.Prediction,
			fmt.Sprintf("%.6f", result.Probabilities[0]),
			fmt.Sprintf("%.6f", result.Probabilities[1]),
			fmt.Sprintf("%.6f", result.Probabilities[2]),
		)
		if err := writer.Write(record); err != nil {
			return err
		}
	}
	writer.Flush()
	if err := writer.Error(); err != nil {
		return err
	}

	return nil

}

func main() {
	modelPath := flag.String("model", "model.json", "Path to the pre-trained model")
	sepalLength := flag.Float64("sepal-length", 0.0, "Sepal length in cm")
	sepalWidth := flag.Float64("sepal-width", 0.0, "Sepal width in cm")
	petalLength := flag.Float64("petal-length", 0.0, "Petal length in cm")
	petalWidth := flag.Float64("petal-width", 0.0, "Petal width in cm")

	batchInput := flag.String("batch-input", "", "Path to input CSV for batch processing")
	batchOutput := flag.String("batch-output", "predictions.csv", "Path to save batch results")
	workers := flag.Int("workers", 1, "Number of concurrent workers for batch mode")
	benchmarkRepeats := flag.Int("benchmark-repeats", 1, "Number of times to run batch for benchmarking")
	flag.Parse()

	model, scaler, artifact, err := irisnn.LoadModel(*modelPath)
	if err != nil {
		log.Fatal(err)
	}

	if *batchInput != "" {
		if err := runBatchMode(model, scaler, artifact.LabelNames, *batchInput, *batchOutput, *workers, *benchmarkRepeats); err != nil {
			log.Fatal(err)
		}
		return
	}

	if *sepalLength > 0 && *sepalWidth > 0 && *petalLength > 0 && *petalWidth > 0 {
		if err := runSingleMode(scaler, model, artifact, *sepalLength, *sepalWidth, *petalLength, *petalWidth); err != nil {
			log.Fatal(err)
		}
		return
	}

	fmt.Println("Provide either --batch-input or all four single-sample measurements.")

}

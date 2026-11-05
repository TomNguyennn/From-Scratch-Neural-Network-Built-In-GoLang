package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"runtime"
	"sync"
	"time"
)

const transactionCSVColumns = 10

type splitJob struct {
	index int
	row   []string
}

type splitResult struct {
	index int
	row   []string
	label string
	err   error
}

func main() {
	inputPath := flag.String("input", "credit-card-fraud-datasets/transactions_train.csv", "Path to the source transactions CSV")
	trainOutput := flag.String("train-output", "credit-card-fraud-datasets/transactions_train_split.csv", "Path to write the training split")
	validationOutput := flag.String("validation-output", "credit-card-fraud-datasets/transactions_validation.csv", "Path to write the validation split")
	testOutput := flag.String("test-output", "credit-card-fraud-datasets/transactions_test.csv", "Path to write the test split")
	validationRatio := flag.Float64("validation-ratio", 0.2, "Fraction of each class written to validation")
	testRatio := flag.Float64("test-ratio", 0.1, "Fraction of each class written to test")
	maxRows := flag.Int("max-rows", 0, "Optional maximum number of rows to process; 0 uses all rows")
	workers := flag.Int("workers", runtime.NumCPU(), "Number of worker goroutines for row validation")
	flag.Parse()

	if *validationRatio <= 0 || *validationRatio >= 1 {
		log.Fatal("validation-ratio must be between 0 and 1")
	}
	if *testRatio <= 0 || *testRatio >= 1 {
		log.Fatal("test-ratio must be between 0 and 1")
	}
	if *validationRatio+*testRatio >= 1 {
		log.Fatal("validation-ratio + test-ratio must be less than 1")
	}
	if *workers < 1 {
		log.Fatal("workers must be at least 1")
	}
	startTime := time.Now()
	trainCount, validationCount, testCount, err := splitTransactionsCSV(*inputPath, *trainOutput, *validationOutput, *testOutput, *validationRatio, *testRatio, *maxRows, *workers)
	if err != nil {
		log.Fatal(err)
	}
	duration := time.Since(startTime)

	fmt.Printf("Wrote %d rows to %s\n", trainCount, *trainOutput)
	fmt.Printf("Wrote %d rows to %s\n", validationCount, *validationOutput)
	fmt.Printf("Wrote %d rows to %s\n", testCount, *testOutput)
	fmt.Printf("Processed rows with %d worker goroutines\n", *workers)
	fmt.Printf("Time took %v\n", duration)
}

func splitTransactionsCSV(inputPath, trainPath, validationPath, testPath string, validationRatio, testRatio float64, maxRows int, workers int) (int, int, int, error) {
	inputFile, err := os.Open(inputPath)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("open input csv: %w", err)
	}
	defer inputFile.Close()

	reader := csv.NewReader(inputFile)
	header, err := reader.Read()
	if err != nil {
		return 0, 0, 0, fmt.Errorf("read header: %w", err)
	}

	trainFile, err := os.Create(trainPath)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("create train split: %w", err)
	}
	defer trainFile.Close()

	validationFile, err := os.Create(validationPath)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("create validation split: %w", err)
	}
	defer validationFile.Close()

	testFile, err := os.Create(testPath)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("create test split: %w", err)
	}
	defer testFile.Close()

	trainWriter := csv.NewWriter(trainFile)
	defer trainWriter.Flush()
	validationWriter := csv.NewWriter(validationFile)
	defer validationWriter.Flush()
	testWriter := csv.NewWriter(testFile)
	defer testWriter.Flush()

	if err := trainWriter.Write(header); err != nil {
		return 0, 0, 0, err
	}
	if err := validationWriter.Write(header); err != nil {
		return 0, 0, 0, err
	}
	if err := testWriter.Write(header); err != nil {
		return 0, 0, 0, err
	}

	seenCounts := make(map[string]int)
	validationCounts := make(map[string]int)
	testCounts := make(map[string]int)
	trainRows := 0
	validationRows := 0
	testRows := 0
	rowIndex := 0
	jobs := make(chan splitJob, workers*2)
	results := make(chan splitResult, workers*2)

	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go splitWorker(jobs, results, &wg)
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	readErrCh := make(chan error, 1)
	go func() {
		defer close(jobs)
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
			jobs <- splitJob{
				index: rowIndex,
				row:   row,
			}
			rowIndex++
		}
	}()

	pending := make(map[int]splitResult)
	nextIndex := 0
	for result := range results {
		if result.err != nil {
			return trainRows, validationRows, testRows, result.err
		}
		pending[result.index] = result

		for {
			current, ok := pending[nextIndex]
			if !ok {
				break
			}
			delete(pending, nextIndex)

			if len(current.row) == 0 {
				nextIndex++
				continue
			}

			destination := chooseTransactionSplit(current.label, validationRatio, testRatio, seenCounts, validationCounts, testCounts)
			switch destination {
			case "validation":
				if err := validationWriter.Write(current.row); err != nil {
					return trainRows, validationRows, testRows, err
				}
				validationRows++
			case "test":
				if err := testWriter.Write(current.row); err != nil {
					return trainRows, validationRows, testRows, err
				}
				testRows++
			default:
				if err := trainWriter.Write(current.row); err != nil {
					return trainRows, validationRows, testRows, err
				}
				trainRows++
			}

			nextIndex++
		}
	}

	if err := <-readErrCh; err != nil {
		return trainRows, validationRows, testRows, err
	}

	if err := trainWriter.Error(); err != nil {
		return trainRows, validationRows, testRows, err
	}
	if err := validationWriter.Error(); err != nil {
		return trainRows, validationRows, testRows, err
	}
	if err := testWriter.Error(); err != nil {
		return trainRows, validationRows, testRows, err
	}

	return trainRows, validationRows, testRows, nil
}

func chooseTransactionSplit(label string, validationRatio, testRatio float64, seenCounts, validationCounts, testCounts map[string]int) string {
	seenCounts[label]++
	targetValidation := int(math.Round(float64(seenCounts[label]) * validationRatio))
	targetTest := int(math.Round(float64(seenCounts[label]) * testRatio))
	if validationCounts[label] < targetValidation {
		validationCounts[label]++
		return "validation"
	}
	if testCounts[label] < targetTest {
		testCounts[label]++
		return "test"
	}
	return "train"
}

func splitWorker(jobs <-chan splitJob, results chan<- splitResult, wg *sync.WaitGroup) {
	defer wg.Done()
	for job := range jobs {
		if len(job.row) == 0 {
			results <- splitResult{index: job.index, row: job.row}
			continue
		}
		if len(job.row) != transactionCSVColumns {
			results <- splitResult{
				index: job.index,
				err:   fmt.Errorf("row %d: expected %d columns, got %d", job.index+2, transactionCSVColumns, len(job.row)),
			}
			continue
		}
		results <- splitResult{
			index: job.index,
			row:   append([]string(nil), job.row...),
			label: job.row[len(job.row)-1],
		}
	}
}

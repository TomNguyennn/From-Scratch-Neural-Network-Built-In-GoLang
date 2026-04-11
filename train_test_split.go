package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"time"
)

// encodedRow holds one processed sample after converting the species label
// into a one-hot encoded target vector.
type encodedRow struct {
	values []string
	label  string
}

func main() {
	// Accept CLI flags so the split script can be reused with different ratios,
	// output files, and random seeds.
	inputPath := flag.String("input", "Iris.csv", "input CSV file")
	trainPath := flag.String("train", "train.csv", "output training CSV file")
	validationPath := flag.String("validation", "validation.csv", "output validation CSV file")
	testPath := flag.String("test", "test.csv", "output test CSV file")
	trainRatio := flag.Float64("train-ratio", 0.7, "fraction of rows per class for training")
	validationRatio := flag.Float64("validation-ratio", 0.15, "fraction of rows per class for validation")
	testRatio := flag.Float64("test-ratio", 0.15, "fraction of rows per class for testing")
	seed := flag.Int64("seed", time.Now().UnixNano(), "random seed for shuffling")
	flag.Parse()

	if err := validateRatios(*trainRatio, *validationRatio, *testRatio); err != nil {
		log.Fatal(err)
	}

	header, rows, err := readAndEncodeRows(*inputPath)
	if err != nil {
		log.Fatal(err)
	}

	// Split by class so each output file keeps a similar class distribution.
	trainRows, validationRows, testRows, err := stratifiedSplit(rows, *trainRatio, *validationRatio, *testRatio, *seed)
	if err != nil {
		log.Fatal(err)
	}

	if err := writeCSV(*trainPath, header, trainRows); err != nil {
		log.Fatalf("write train csv: %v", err)
	}
	if err := writeCSV(*validationPath, header, validationRows); err != nil {
		log.Fatalf("write validation csv: %v", err)
	}
	if err := writeCSV(*testPath, header, testRows); err != nil {
		log.Fatalf("write test csv: %v", err)
	}

	fmt.Printf("Created %s with %d rows\n", *trainPath, len(trainRows))
	fmt.Printf("Created %s with %d rows\n", *validationPath, len(validationRows))
	fmt.Printf("Created %s with %d rows\n", *testPath, len(testRows))
}

func validateRatios(trainRatio, validationRatio, testRatio float64) error {
	// All three ratios must be positive and together form a complete dataset split.
	if trainRatio <= 0 || validationRatio <= 0 || testRatio <= 0 {
		return fmt.Errorf("all split ratios must be greater than 0")
	}

	total := trainRatio + validationRatio + testRatio
	if math.Abs(total-1.0) > 1e-9 {
		return fmt.Errorf("split ratios must add up to 1.0; got %.6f", total)
	}

	return nil
}

func readAndEncodeRows(path string) ([]string, []encodedRow, error) {
	// Read the raw Iris CSV and convert the species column into three output
	// columns so the neural network can learn a multiclass target.
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("open input csv: %w", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, nil, fmt.Errorf("read input csv: %w", err)
	}
	if len(records) < 2 {
		return nil, nil, fmt.Errorf("input csv must contain a header and at least one data row")
	}

	header := []string{
		"Id",
		"SepalLengthCm",
		"SepalWidthCm",
		"PetalLengthCm",
		"PetalWidthCm",
		"Setosa",
		"Versicolor",
		"Virginica",
	}

	rows := make([]encodedRow, 0, len(records)-1)
	for i, record := range records[1:] {
		if len(record) != 6 {
			return nil, nil, fmt.Errorf("row %d has %d columns, expected 6", i+2, len(record))
		}

		encoded, label, err := encodeSpecies(record[5])
		if err != nil {
			return nil, nil, fmt.Errorf("row %d: %w", i+2, err)
		}

		values := append([]string{}, record[:5]...)
		values = append(values, encoded...)
		rows = append(rows, encodedRow{
			values: values,
			label:  label,
		})
	}

	return header, rows, nil
}

func encodeSpecies(species string) ([]string, string, error) {
	// One-hot encoding turns the flower class into the 3 output neurons that
	// the classifier will predict.
	switch species {
	case "Iris-setosa":
		return []string{"1", "0", "0"}, species, nil
	case "Iris-versicolor":
		return []string{"0", "1", "0"}, species, nil
	case "Iris-virginica":
		return []string{"0", "0", "1"}, species, nil
	default:
		return nil, "", fmt.Errorf("unknown species %q", species)
	}
}

func stratifiedSplit(rows []encodedRow, trainRatio, validationRatio, testRatio float64, seed int64) ([][]string, [][]string, [][]string, error) {
	// Group rows by label before shuffling so each split gets examples from
	// every flower class.
	grouped := make(map[string][]encodedRow)
	for _, row := range rows {
		grouped[row.label] = append(grouped[row.label], row)
	}

	rng := rand.New(rand.NewSource(seed))
	trainRows := make([][]string, 0, len(rows))
	validationRows := make([][]string, 0, len(rows))
	testRows := make([][]string, 0, len(rows))

	for label, group := range grouped {
		if len(group) < 3 {
			return nil, nil, nil, fmt.Errorf("label %q needs at least 3 rows for train/validation/test splits", label)
		}

		// Shuffle inside each class bucket before slicing into train, validation,
		// and test sets.
		rng.Shuffle(len(group), func(i, j int) {
			group[i], group[j] = group[j], group[i]
		})

		trainCount := int(math.Round(float64(len(group)) * trainRatio))
		validationCount := int(math.Round(float64(len(group)) * validationRatio))
		if trainCount < 1 {
			trainCount = 1
		}
		if validationCount < 1 {
			validationCount = 1
		}
		if trainCount+validationCount >= len(group) {
			validationCount = 1
			trainCount = len(group) - 2
		}
		testCount := len(group) - trainCount - validationCount
		if testCount < 1 {
			return nil, nil, nil, fmt.Errorf("label %q does not leave any rows for test split", label)
		}

		for _, row := range group[:trainCount] {
			trainRows = append(trainRows, row.values)
		}
		for _, row := range group[trainCount : trainCount+validationCount] {
			validationRows = append(validationRows, row.values)
		}
		for _, row := range group[trainCount+validationCount:] {
			testRows = append(testRows, row.values)
		}
	}

	rng.Shuffle(len(trainRows), func(i, j int) {
		trainRows[i], trainRows[j] = trainRows[j], trainRows[i]
	})
	rng.Shuffle(len(validationRows), func(i, j int) {
		validationRows[i], validationRows[j] = validationRows[j], validationRows[i]
	})
	rng.Shuffle(len(testRows), func(i, j int) {
		testRows[i], testRows[j] = testRows[j], testRows[i]
	})

	return trainRows, validationRows, testRows, nil
}

func writeCSV(path string, header []string, rows [][]string) (err error) {
	// Write the transformed dataset back to disk with the header preserved.
	file, err := os.Create(path)
	if err != nil {
		return err
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
	if err := writer.WriteAll(rows); err != nil {
		return err
	}

	writer.Flush()
	if err := writer.Error(); err != nil {
		return err
	}

	return nil
}

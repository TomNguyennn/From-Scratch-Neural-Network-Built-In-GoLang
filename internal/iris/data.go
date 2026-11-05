package iris

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"

	"go_neural_network/internal/irisnn"
	"gonum.org/v1/gonum/mat"
)

const (
	InputFeatures = 4
	OutputClasses = 3
)

func DefaultLabelNames() map[int]string {
	return map[int]string{
		0: "Setosa",
		1: "Versicolor",
		2: "Virginica",
	}
}

func LoadDataset(path string) (*irisnn.Dataset, error) {
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
	inputsData := make([]float64, samples*InputFeatures)
	labelsData := make([]float64, samples*OutputClasses)

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

	return irisnn.NewDataset(
		mat.NewDense(samples, InputFeatures, inputsData),
		mat.NewDense(samples, OutputClasses, labelsData),
	)
}

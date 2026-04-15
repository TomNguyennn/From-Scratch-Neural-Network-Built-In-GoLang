package irisnn

import (
	"fmt"
	"math"
	"os"
	"encoding/csv"
	"strconv"
	"gonum.org/v1/gonum/mat"
)
// dataset keeps one split of the Iris data in matrix form.
type Dataset struct {
	Inputs *mat.Dense
	Labels *mat.Dense
	Rows   int
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



func LoadDataset(path string) (*Dataset, error) {
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

	return &Dataset{
		Inputs: mat.NewDense(samples, 4, inputsData),
		Labels: mat.NewDense(samples, 3, labelsData),
		Rows:   samples,
	}, nil
}

func FitFeatureScaler(inputs *mat.Dense) *FeatureScaler {
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

	return &FeatureScaler{Mean: mean, Std: std}
}

func (s *FeatureScaler) Transform(inputs *mat.Dense) *mat.Dense {
	rows, cols := inputs.Dims()
	out := mat.NewDense(rows, cols, nil)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			out.Set(r, c, (inputs.At(r, c)-s.Mean[c])/s.Std[c])
		}
	}
	return out
}


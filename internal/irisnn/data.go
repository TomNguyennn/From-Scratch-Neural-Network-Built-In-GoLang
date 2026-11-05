package irisnn

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

// Dataset keeps one dataset split in matrix form.
type Dataset struct {
	Inputs *mat.Dense
	Labels *mat.Dense
	Rows   int
}

func NewDataset(inputs, labels *mat.Dense) (*Dataset, error) {
	if inputs == nil {
		return nil, fmt.Errorf("inputs matrix is nil")
	}
	if labels == nil {
		return nil, fmt.Errorf("labels matrix is nil")
	}

	rows, inputCols := inputs.Dims()
	labelRows, labelCols := labels.Dims()
	if rows == 0 || labelRows == 0 {
		return nil, fmt.Errorf("dataset is empty")
	}
	if rows != labelRows {
		return nil, fmt.Errorf("input rows (%d) do not match label rows (%d)", rows, labelRows)
	}
	if inputCols == 0 {
		return nil, fmt.Errorf("dataset inputs must have at least one column")
	}
	if labelCols == 0 {
		return nil, fmt.Errorf("dataset labels must have at least one column")
	}

	return &Dataset{
		Inputs: inputs,
		Labels: labels,
		Rows:   rows,
	}, nil
}

func NewDatasetFromRaw(inputsData []float64, rows, inputWidth int, labelsData []float64, outputWidth int) (*Dataset, error) {
	if rows <= 0 {
		return nil, fmt.Errorf("rows must be greater than 0")
	}
	if inputWidth <= 0 {
		return nil, fmt.Errorf("input width must be greater than 0")
	}
	if outputWidth <= 0 {
		return nil, fmt.Errorf("output width must be greater than 0")
	}
	if len(inputsData) != rows*inputWidth {
		return nil, fmt.Errorf("got %d input values, want %d", len(inputsData), rows*inputWidth)
	}
	if len(labelsData) != rows*outputWidth {
		return nil, fmt.Errorf("got %d label values, want %d", len(labelsData), rows*outputWidth)
	}

	return NewDataset(
		mat.NewDense(rows, inputWidth, inputsData),
		mat.NewDense(rows, outputWidth, labelsData),
	)
}

func BinaryLabelsToOneHot(labels []int) (*mat.Dense, error) {
	if len(labels) == 0 {
		return nil, fmt.Errorf("binary labels are empty")
	}

	data := make([]float64, len(labels)*2)
	for i, label := range labels {
		switch label {
		case 0:
			data[i*2] = 1
		case 1:
			data[i*2+1] = 1
		default:
			return nil, fmt.Errorf("binary label at row %d must be 0 or 1, got %d", i, label)
		}
	}

	return mat.NewDense(len(labels), 2, data), nil
}

func NewBinaryDataset(inputs *mat.Dense, labels []int) (*Dataset, error) {
	encodedLabels, err := BinaryLabelsToOneHot(labels)
	if err != nil {
		return nil, err
	}
	return NewDataset(inputs, encodedLabels)
}

type FeatureScaler struct {
	Mean []float64 `json:"mean"`
	Std  []float64 `json:"std"`
}

func FitFeatureScaler(inputs *mat.Dense) *FeatureScaler {
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

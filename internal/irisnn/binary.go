package irisnn

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

type BinaryMetrics struct {
	Loss      float64 `json:"loss"`
	Accuracy  float64 `json:"accuracy"`
	Precision float64 `json:"precision"`
	Recall    float64 `json:"recall"`
	F1        float64 `json:"f1"`
	F2        float64 `json:"f2"`
	PosRate   float64 `json:"positive_rate"`
	Threshold float64 `json:"threshold"`
	Rows      int64   `json:"rows"`
	TruePos   int64   `json:"true_pos"`
	FalsePos  int64   `json:"false_pos"`
	TrueNeg   int64   `json:"true_neg"`
	FalseNeg  int64   `json:"false_neg"`
}

func BinaryPositiveClassProbabilities(predictions *mat.Dense) ([]float64, error) {
	return PositiveClassProbabilities(predictions, 1)
}

func PositiveClassProbabilities(predictions *mat.Dense, positiveClass int) ([]float64, error) {
	if predictions == nil {
		return nil, fmt.Errorf("predictions matrix is nil")
	}

	rows, cols := predictions.Dims()
	if rows == 0 {
		return nil, fmt.Errorf("predictions matrix is empty")
	}
	if positiveClass < 0 || positiveClass >= cols {
		return nil, fmt.Errorf("positive class index %d out of range for %d columns", positiveClass, cols)
	}

	probabilities := make([]float64, rows)
	for r := 0; r < rows; r++ {
		probabilities[r] = predictions.At(r, positiveClass)
	}
	return probabilities, nil
}

func BinaryActualLabels(labels *mat.Dense) ([]int, error) {
	if labels == nil {
		return nil, fmt.Errorf("labels matrix is nil")
	}

	rows, cols := labels.Dims()
	if rows == 0 {
		return nil, fmt.Errorf("labels matrix is empty")
	}
	if cols != 2 {
		return nil, fmt.Errorf("binary labels must have 2 output columns, got %d", cols)
	}

	actuals := make([]int, rows)
	for r := 0; r < rows; r++ {
		actuals[r] = argmaxRow(labels, r)
	}
	return actuals, nil
}

func EvaluateBinaryThreshold(probabilities []float64, actuals []int, threshold float64) (BinaryMetrics, error) {
	if len(probabilities) == 0 {
		return BinaryMetrics{}, fmt.Errorf("probabilities are empty")
	}
	if len(probabilities) != len(actuals) {
		return BinaryMetrics{}, fmt.Errorf("probabilities length (%d) does not match actual labels length (%d)", len(probabilities), len(actuals))
	}
	if threshold < 0 || threshold > 1 {
		return BinaryMetrics{}, fmt.Errorf("threshold must be between 0 and 1, got %f", threshold)
	}

	metrics := BinaryMetrics{Threshold: threshold}
	for i, probability := range probabilities {
		if probability < 0 || probability > 1 {
			return BinaryMetrics{}, fmt.Errorf("probability at row %d must be between 0 and 1, got %f", i, probability)
		}
		actual := actuals[i]
		if actual != 0 && actual != 1 {
			return BinaryMetrics{}, fmt.Errorf("actual label at row %d must be 0 or 1, got %d", i, actual)
		}

		predicted := 0
		if probability >= threshold {
			predicted = 1
		}

		metrics.Rows++
		if actual == 1 {
			metrics.PosRate++
		}

		clipped := math.Min(math.Max(probability, 1e-9), 1-1e-9)
		metrics.Loss += -(float64(actual)*math.Log(clipped) + float64(1-actual)*math.Log(1-clipped))

		switch {
		case predicted == 1 && actual == 1:
			metrics.TruePos++
		case predicted == 1 && actual == 0:
			metrics.FalsePos++
		case predicted == 0 && actual == 0:
			metrics.TrueNeg++
		case predicted == 0 && actual == 1:
			metrics.FalseNeg++
		}
	}

	if metrics.Rows > 0 {
		metrics.Loss /= float64(metrics.Rows)
		metrics.Accuracy = float64(metrics.TruePos+metrics.TrueNeg) / float64(metrics.Rows)
		metrics.PosRate /= float64(metrics.Rows)

		if denom := float64(metrics.TruePos + metrics.FalsePos); denom > 0 {
			metrics.Precision = float64(metrics.TruePos) / denom
		}
		if denom := float64(metrics.TruePos + metrics.FalseNeg); denom > 0 {
			metrics.Recall = float64(metrics.TruePos) / denom
		}
		if metrics.Precision+metrics.Recall > 0 {
			metrics.F1 = 2 * metrics.Precision * metrics.Recall / (metrics.Precision + metrics.Recall)
		}
		if denom := 4*metrics.Precision + metrics.Recall; denom > 0 {
			metrics.F2 = 5 * metrics.Precision * metrics.Recall / denom
		}
	}

	return metrics, nil
}

func EvaluateBinaryOutputs(predictions, labels *mat.Dense, threshold float64) (BinaryMetrics, error) {
	probabilities, err := BinaryPositiveClassProbabilities(predictions)
	if err != nil {
		return BinaryMetrics{}, err
	}

	actuals, err := BinaryActualLabels(labels)
	if err != nil {
		return BinaryMetrics{}, err
	}

	return EvaluateBinaryThreshold(probabilities, actuals, threshold)
}

package irisnn

import (
	"fmt"
	"math"
	"sort"
)

// BestF2Threshold evaluates every distinct score boundary in O(N log N) time
// and O(N) extra memory. Inputs are not modified; predictions use score >= threshold.
func BestF2Threshold(probabilities []float64, actuals []int) (BinaryMetrics, error) {
	if len(probabilities) == 0 || len(probabilities) != len(actuals) {
		return BinaryMetrics{}, fmt.Errorf("probabilities and labels must have equal nonzero lengths")
	}
	type observation struct {
		score float64
		label int
	}
	ranked := make([]observation, len(probabilities))
	positives := 0
	for i, p := range probabilities {
		if math.IsNaN(p) || math.IsInf(p, 0) || p < 0 || p > 1 {
			return BinaryMetrics{}, fmt.Errorf("invalid probability at row %d", i)
		}
		if actuals[i] != 0 && actuals[i] != 1 {
			return BinaryMetrics{}, fmt.Errorf("invalid binary label at row %d", i)
		}
		ranked[i] = observation{p, actuals[i]}
		positives += actuals[i]
	}
	if positives == 0 {
		return BinaryMetrics{}, fmt.Errorf("threshold tuning requires at least one positive label")
	}
	sort.Slice(ranked, func(i, j int) bool { return ranked[i].score > ranked[j].score })
	best := BinaryMetrics{F2: -1}
	tp, fp := 0, 0
	for i := 0; i < len(ranked); {
		j := i
		for j < len(ranked) && ranked[j].score == ranked[i].score {
			tp += ranked[j].label
			fp += 1 - ranked[j].label
			j++
		}
		current := BinaryMetrics{
			Threshold: ranked[i].score,
			F2:        5 * float64(tp) / (5*float64(tp) + 4*float64(positives-tp) + float64(fp)),
			Recall:    float64(tp) / float64(positives),
			Precision: float64(tp) / float64(tp+fp),
		}
		if betterF2Candidate(current, best) {
			best = current
		}
		i = j
	}
	return EvaluateBinaryThreshold(probabilities, actuals, best.Threshold)
}

func betterF2Candidate(current, best BinaryMetrics) bool {
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
			if math.Abs(current.Precision-best.Precision) <= 1e-12 && current.Threshold < best.Threshold {
				return true
			}
		}
	}
	return false
}

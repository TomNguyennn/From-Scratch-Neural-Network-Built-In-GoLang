package irisnn

import (
	"fmt"
	"math"
	"sort"
)

// ProbabilityStats summarizes predicted fraud probabilities within one true class.
type ProbabilityStats struct {
	Count  int     `json:"count"`
	Mean   float64 `json:"mean"`
	Median float64 `json:"median"`
	P90    float64 `json:"p90"`
	P99    float64 `json:"p99"`
}

type ClassProbabilityStats struct {
	Legitimate *ProbabilityStats `json:"legitimate"`
	Fraud      *ProbabilityStats `json:"fraud"`
}

// SummarizeClassProbabilities uses linear interpolation at (n-1)*quantile.
// It does not mutate inputs. An absent class is represented by nil.
func SummarizeClassProbabilities(probabilities []float64, actuals []int) (ClassProbabilityStats, error) {
	if len(probabilities) != len(actuals) || len(probabilities) == 0 {
		return ClassProbabilityStats{}, fmt.Errorf("probabilities and labels must have equal nonzero lengths")
	}
	var groups [2][]float64
	for i, p := range probabilities {
		if math.IsNaN(p) || math.IsInf(p, 0) || p < 0 || p > 1 {
			return ClassProbabilityStats{}, fmt.Errorf("invalid probability at row %d", i)
		}
		if actuals[i] != 0 && actuals[i] != 1 {
			return ClassProbabilityStats{}, fmt.Errorf("invalid binary label at row %d", i)
		}
		groups[actuals[i]] = append(groups[actuals[i]], p)
	}
	summarize := func(scores []float64) *ProbabilityStats {
		if len(scores) == 0 {
			return nil
		}
		sum := 0.0
		for _, p := range scores {
			sum += p
		}
		sort.Float64s(scores)
		percentile := func(q float64) float64 {
			position := float64(len(scores)-1) * q
			low := int(position)
			high := min(low+1, len(scores)-1)
			return scores[low] + (scores[high]-scores[low])*(position-float64(low))
		}
		return &ProbabilityStats{len(scores), sum / float64(len(scores)), percentile(.5), percentile(.9), percentile(.99)}
	}
	return ClassProbabilityStats{summarize(groups[0]), summarize(groups[1])}, nil
}

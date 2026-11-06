package irisnn

import (
	"fmt"
	"math"
	"sort"
)

// AveragePrecision sums precision weighted by recall increases, grouping tied
// scores. No positive labels yields 0. Inputs are not modified.
func AveragePrecision(probabilities []float64, actuals []int) (float64, error) {
	if len(probabilities) == 0 || len(probabilities) != len(actuals) {
		return 0, fmt.Errorf("probabilities and labels must have equal nonzero lengths")
	}
	type observation struct {
		score float64
		label int
	}
	ranked := make([]observation, len(probabilities))
	positives := 0
	for i, p := range probabilities {
		if math.IsNaN(p) || math.IsInf(p, 0) || p < 0 || p > 1 {
			return 0, fmt.Errorf("invalid probability at row %d", i)
		}
		if actuals[i] != 0 && actuals[i] != 1 {
			return 0, fmt.Errorf("invalid binary label at row %d", i)
		}
		ranked[i] = observation{p, actuals[i]}
		positives += actuals[i]
	}
	if positives == 0 {
		return 0, nil
	}
	sort.Slice(ranked, func(i, j int) bool { return ranked[i].score > ranked[j].score })
	tp := 0
	ap := 0.0
	for i := 0; i < len(ranked); {
		j := i
		added := 0
		for j < len(ranked) && ranked[j].score == ranked[i].score {
			added += ranked[j].label
			j++
		}
		tp += added
		ap += float64(added) / float64(positives) * float64(tp) / float64(j)
		i = j
	}
	return ap, nil
}

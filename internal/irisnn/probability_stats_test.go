package irisnn

import (
	"math"
	"reflect"
	"testing"
)

func TestClassProbabilityStats(t *testing.T) {
	p := []float64{1, .2, 0, .8, .5}
	original := append([]float64(nil), p...)
	got, err := SummarizeClassProbabilities(p, []int{0, 1, 0, 1, 0})
	if err != nil {
		t.Fatal(err)
	}
	if *got.Legitimate != (ProbabilityStats{3, .5, .5, .9, .99}) {
		t.Fatal(got.Legitimate)
	}
	if got.Fraud.Count != 2 || math.Abs(got.Fraud.P90-.74) > 1e-12 || math.Abs(got.Fraud.P99-.794) > 1e-12 || got.Fraud.Mean != .5 || got.Fraud.Median != .5 {
		t.Fatal(got.Fraud)
	}
	if !reflect.DeepEqual(p, original) {
		t.Fatal("mutated inputs")
	}
	single, err := SummarizeClassProbabilities([]float64{.4}, []int{1})
	if err != nil || single.Legitimate != nil || *single.Fraud != (ProbabilityStats{1, .4, .4, .4, .4}) {
		t.Fatal(single, err)
	}
	for _, p := range []float64{math.NaN(), math.Inf(1), -.1, 1.1} {
		if _, err := SummarizeClassProbabilities([]float64{p}, []int{0}); err == nil {
			t.Fatal("accepted invalid probability")
		}
	}
	for _, ys := range [][]int{nil, {2}, {-1}} {
		if _, err := SummarizeClassProbabilities([]float64{.5}, ys); err == nil {
			t.Fatal("accepted invalid labels")
		}
	}
	if _, err := SummarizeClassProbabilities(nil, nil); err == nil {
		t.Fatal("accepted empty input")
	}
}

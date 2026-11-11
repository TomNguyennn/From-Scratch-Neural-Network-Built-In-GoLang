package irisnn

import (
	"math"
	"math/rand"
	"reflect"
	"testing"
)

func TestBestF2Threshold(t *testing.T) {
	cases := []struct {
		p    []float64
		y    []int
		want float64
	}{
		{[]float64{.8, .8, .2}, []int{1, 0, 1}, .2},
		{[]float64{.5, .5, .5}, []int{0, 1, 0}, .5},
		{[]float64{.9, .3, .1}, []int{1, 1, 1}, .1},
		{[]float64{0, 1}, []int{1, 0}, 0},
		{[]float64{0, 1}, []int{0, 1}, 1},
		{[]float64{.5, math.Nextafter(.5, 1)}, []int{0, 1}, math.Nextafter(.5, 1)},
		{[]float64{.502, .501}, []int{1, 0}, .502},
		// Equal F2 at .9 and .1: recall decides in favor of .1.
		{[]float64{.9, .8, .7, .6, .5, .4, .3, .2, .1}, []int{1, 0, 0, 0, 0, 0, 0, 0, 1}, .1},
	}
	check := func(p []float64, y []int) BinaryMetrics {
		t.Helper()
		pc, yc := append([]float64(nil), p...), append([]int(nil), y...)
		got, err := BestF2Threshold(p, y)
		if err != nil {
			t.Fatal(err)
		}
		best := BinaryMetrics{F2: -1}
		for _, threshold := range p {
			m, err := EvaluateBinaryThreshold(p, y, threshold)
			if err != nil {
				t.Fatal(err)
			}
			// Independent lexicographic reference, with the documented tolerance.
			a, b := []float64{m.F2, m.Recall, m.Precision, -m.Threshold}, []float64{best.F2, best.Recall, best.Precision, -best.Threshold}
			for k := range a {
				tolerance := 1e-12
				if k == 3 {
					tolerance = 0
				}
				if math.Abs(a[k]-b[k]) <= tolerance {
					continue
				}
				if a[k] > b[k] {
					best = m
				}
				break
			}
		}
		if got != best {
			t.Fatalf("got %+v want %+v", got, best)
		}
		evaluated, _ := EvaluateBinaryThreshold(p, y, got.Threshold)
		if got != evaluated {
			t.Fatal("evaluation mismatch")
		}
		if !reflect.DeepEqual(p, pc) || !reflect.DeepEqual(y, yc) {
			t.Fatal("inputs modified")
		}
		return got
	}
	for _, tc := range cases {
		if got := check(tc.p, tc.y); got.Threshold != tc.want {
			t.Fatalf("threshold %g want %g", got.Threshold, tc.want)
		}
	}
	rng := rand.New(rand.NewSource(42))
	for n := 1; n <= 100; n++ {
		p, y := make([]float64, n), make([]int, n)
		for i := range p {
			p[i] = float64(rng.Intn(11)) / 10
			y[i] = rng.Intn(2)
		}
		y[0] = 1
		check(p, y)
	}
}

func TestBestF2ThresholdErrors(t *testing.T) {
	cases := []struct {
		p []float64
		y []int
	}{
		{nil, nil}, {[]float64{.5}, nil}, {nil, []int{1}},
		{[]float64{.5}, []int{0}}, {[]float64{.5}, []int{2}}, {[]float64{.5}, []int{-1}},
	}
	for _, p := range []float64{math.NaN(), math.Inf(1), math.Inf(-1), -.1, 1.1} {
		cases = append(cases, struct {
			p []float64
			y []int
		}{[]float64{p}, []int{1}})
	}
	for _, tc := range cases {
		if _, err := BestF2Threshold(tc.p, tc.y); err == nil {
			t.Fatalf("accepted %v %v", tc.p, tc.y)
		}
	}
}

func TestF2CandidateTieBreaking(t *testing.T) {
	best := BinaryMetrics{F2: .5, Recall: .5, Precision: .5, Threshold: .5}
	for _, current := range []BinaryMetrics{
		{F2: .5 + 5e-13, Recall: .6, Precision: .1, Threshold: .9},
		{F2: .5, Recall: .5, Precision: .6, Threshold: .9},
		{F2: .5, Recall: .5, Precision: .5, Threshold: .4},
	} {
		if !betterF2Candidate(current, best) || betterF2Candidate(best, current) {
			t.Fatalf("tie comparison: %+v", current)
		}
	}
}

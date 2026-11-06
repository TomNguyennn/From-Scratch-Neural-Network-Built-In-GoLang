package irisnn

import (
	"math"
	"reflect"
	"testing"
)

func TestAveragePrecision(t *testing.T) {
	for _, tc := range []struct {
		name string
		p    []float64
		y    []int
		want float64
	}{
		{"perfect", []float64{.9, .8, .2}, []int{1, 1, 0}, 1},
		{"reversed", []float64{.9, .8, .2}, []int{0, 0, 1}, 1.0 / 3},
		{"mixed", []float64{.9, .8, .7, .6}, []int{1, 0, 1, 0}, 5.0 / 6},
		{"tied", []float64{.5, .5, .5}, []int{1, 0, 0}, 1.0 / 3},
		{"tied shuffled", []float64{.5, .5, .5}, []int{0, 0, 1}, 1.0 / 3},
		{"groups", []float64{.9, .8, .8}, []int{1, 0, 1}, 5.0 / 6},
		{"no positives", []float64{0, 1}, []int{0, 0}, 0},
		{"all positives", []float64{0, 1}, []int{1, 1}, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := append([]float64(nil), tc.p...)
			y := append([]int(nil), tc.y...)
			got, err := AveragePrecision(p, y)
			if err != nil || math.Abs(got-tc.want) > 1e-12 {
				t.Fatalf("AP %g want %g: %v", got, tc.want, err)
			}
			if !reflect.DeepEqual(p, tc.p) || !reflect.DeepEqual(y, tc.y) {
				t.Fatal("inputs mutated")
			}
		})
	}
	for _, tc := range []struct {
		p []float64
		y []int
	}{
		{nil, nil}, {[]float64{.5}, nil}, {[]float64{.5}, []int{2}},
		{[]float64{math.NaN()}, []int{0}}, {[]float64{math.Inf(1)}, []int{1}},
		{[]float64{-.1}, []int{1}}, {[]float64{1.1}, []int{0}},
	} {
		if _, err := AveragePrecision(tc.p, tc.y); err == nil {
			t.Fatalf("accepted invalid input %+v", tc)
		}
	}
}

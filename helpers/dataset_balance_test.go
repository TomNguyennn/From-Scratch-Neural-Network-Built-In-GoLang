package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCalculateFraudStats(t *testing.T) {
	for _, tc := range []struct {
		name, csv string
		want      FraudStats
		bad       bool
	}{
		{"mixed", "id,isFraud\na,0\nb,1\nc,0\nd,0\n", FraudStats{Total: 4, Fraud: 1, NonFraud: 3, FraudPercent: 25, NonFraudPercent: 75}, false},
		{"all fraud", "isFraud\n1\n1\n", FraudStats{Total: 2, Fraud: 2, FraudPercent: 100}, false},
		{"no fraud", "isFraud\n0\n", FraudStats{Total: 1, NonFraud: 1, NonFraudPercent: 100}, false},
		{"missing column", "label\n1\n", FraudStats{}, true},
		{"duplicate column", "isFraud,isFraud\n0,1\n", FraudStats{}, true},
		{"invalid label", "isFraud\n2\n", FraudStats{}, true},
		{"malformed row", "id,isFraud\nx\n", FraudStats{}, true},
		{"header only", "isFraud\n", FraudStats{}, true},
		{"empty", "", FraudStats{}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "data.csv")
			if err := os.WriteFile(path, []byte(tc.csv), 0600); err != nil {
				t.Fatal(err)
			}
			got, err := CalculateFraudStats(path)
			if tc.bad {
				if err == nil {
					t.Fatal("expected error")
				}
				return
			}
			if err != nil || got != tc.want {
				t.Fatalf("got %+v, %v; want %+v", got, err, tc.want)
			}
		})
	}
	if _, err := CalculateFraudStats(filepath.Join(t.TempDir(), "missing.csv")); err == nil {
		t.Fatal("expected missing file error")
	}
}

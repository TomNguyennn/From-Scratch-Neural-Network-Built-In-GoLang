// Run with: go run ./helpers
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
)

type FraudStats struct {
	Total           int64
	Fraud           int64
	NonFraud        int64
	FraudPercent    float64
	NonFraudPercent float64
}

// CalculateFraudStats streams a CSV whose isFraud column contains 0 or 1.
// An empty dataset is an error because its class percentages are undefined.
func CalculateFraudStats(path string) (FraudStats, error) {
	file, err := os.Open(path)
	if err != nil {
		return FraudStats{}, err
	}
	defer file.Close()
	reader := csv.NewReader(file)
	header, err := reader.Read()
	if err != nil {
		return FraudStats{}, fmt.Errorf("read header: %w", err)
	}
	labelColumn := -1
	for i, name := range header {
		if name == "isFraud" {
			if labelColumn != -1 {
				return FraudStats{}, fmt.Errorf("duplicate isFraud column")
			}
			labelColumn = i
		}
	}
	if labelColumn == -1 {
		return FraudStats{}, fmt.Errorf("missing isFraud column")
	}
	var stats FraudStats
	for row := 2; ; row++ {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return FraudStats{}, fmt.Errorf("row %d: %w", row, err)
		}
		switch record[labelColumn] {
		case "0":
			stats.NonFraud++
		case "1":
			stats.Fraud++
		default:
			return FraudStats{}, fmt.Errorf("row %d: isFraud must be 0 or 1, got %q", row, record[labelColumn])
		}
		stats.Total++
	}
	if stats.Total == 0 {
		return FraudStats{}, fmt.Errorf("dataset has no data rows")
	}
	stats.FraudPercent = 100 * float64(stats.Fraud) / float64(stats.Total)
	stats.NonFraudPercent = 100 * float64(stats.NonFraud) / float64(stats.Total)
	return stats, nil
}

func main() {
	train := flag.String("train-input", "credit-card-fraud-datasets/transactions_train_split.csv", "Training CSV")
	validation := flag.String("validation-input", "credit-card-fraud-datasets/transactions_validation.csv", "Validation CSV")
	test := flag.String("test-input", "credit-card-fraud-datasets/transactions_test.csv", "Test CSV")
	flag.Parse()
	splits := []struct {
		name, path string
		stats      FraudStats
	}{
		{name: "Train", path: *train}, {name: "Validation", path: *validation}, {name: "Test", path: *test},
	}
	for i := range splits {
		stats, err := CalculateFraudStats(splits[i].path)
		if err != nil {
			log.Fatalf("%s (%s): %v", splits[i].name, splits[i].path, err)
		}
		splits[i].stats = stats
	}
	fmt.Printf("%-12s %12s %12s %12s %12s %14s\n", "Split", "Total", "Fraud", "Fraud (%)", "Non-fraud", "Non-fraud (%)")
	for _, split := range splits {
		s := split.stats
		fmt.Printf("%-12s %12d %12d %11.4f%% %12d %13.4f%%\n", split.name, s.Total, s.Fraud, s.FraudPercent, s.NonFraud, s.NonFraudPercent)
	}
}

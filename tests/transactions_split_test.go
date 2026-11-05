package tests

import (
	"path/filepath"
	"testing"
)

func TestTransactionsSplitCreatesThreeOutputs(t *testing.T) {
	tempDir := t.TempDir()
	inputPath := filepath.Join(tempDir, "transactions.csv")
	trainPath := filepath.Join(tempDir, "train.csv")
	validationPath := filepath.Join(tempDir, "validation.csv")
	testPath := filepath.Join(tempDir, "test.csv")

	writeCSVFile(t, inputPath, [][]string{
		{"step", "type", "amount", "nameOrig", "oldbalanceOrig", "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud"},
		{"1", "PAYMENT", "10", "C1", "100", "90", "M1", "0", "0", "0"},
		{"2", "TRANSFER", "500", "C2", "500", "0", "C9", "0", "0", "1"},
		{"3", "PAYMENT", "15", "C3", "100", "85", "M2", "0", "0", "0"},
		{"4", "CASH_OUT", "700", "C4", "700", "0", "C10", "0", "700", "1"},
		{"5", "PAYMENT", "20", "C5", "100", "80", "M3", "0", "0", "0"},
		{"6", "TRANSFER", "600", "C6", "600", "0", "C11", "0", "0", "1"},
		{"7", "PAYMENT", "30", "C7", "100", "70", "M4", "0", "0", "0"},
		{"8", "TRANSFER", "800", "C8", "800", "0", "C12", "0", "0", "1"},
	})

	runGoCommand(
		t,
		"run", "transactions_split.go",
		"--input", inputPath,
		"--train-output", trainPath,
		"--validation-output", validationPath,
		"--test-output", testPath,
		"--validation-ratio", "0.25",
		"--test-ratio", "0.25",
		"--workers", "2",
	)

	trainRows := readCSVFile(t, trainPath)
	validationRows := readCSVFile(t, validationPath)
	testRows := readCSVFile(t, testPath)

	if len(trainRows) <= 1 || len(validationRows) <= 1 || len(testRows) <= 1 {
		t.Fatalf("expected all splits to contain data rows, got train=%d validation=%d test=%d", len(trainRows)-1, len(validationRows)-1, len(testRows)-1)
	}
}

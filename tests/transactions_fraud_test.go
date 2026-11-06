package tests

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func TestTransactionsFraudTrainAndPredictCSV(t *testing.T) {
	for _, program := range []string{"transactions_fraud.go", "transactions_fraud_nn.go"} {
		t.Run(program, func(t *testing.T) { testFraudCLI(t, program) })
	}
}

func testFraudCLI(t *testing.T, program string) {
	tempDir := t.TempDir()
	trainPath := filepath.Join(tempDir, "train.csv")
	validationPath := filepath.Join(tempDir, "validation.csv")
	testPath := filepath.Join(tempDir, "test.csv")
	modelPath := filepath.Join(tempDir, "model.json")
	outputPath := filepath.Join(tempDir, "predictions.csv")

	writeCSVFile(t, trainPath, [][]string{
		{"step", "type", "amount", "nameOrig", "oldbalanceOrig", "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud"},
		{"1", "PAYMENT", "100.0", "C1", "1000.0", "900.0", "M1", "0.0", "0.0", "0"},
		{"2", "TRANSFER", "5000.0", "C2", "5000.0", "0.0", "C9", "0.0", "0.0", "1"},
		{"3", "CASH_OUT", "3000.0", "C3", "3000.0", "0.0", "C10", "100.0", "3100.0", "1"},
		{"4", "PAYMENT", "150.0", "C4", "1000.0", "850.0", "M2", "0.0", "0.0", "0"},
	})
	writeCSVFile(t, validationPath, [][]string{
		{"step", "type", "amount", "nameOrig", "oldbalanceOrig", "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud"},
		{"5", "TRANSFER", "4000.0", "C5", "4000.0", "0.0", "C11", "0.0", "0.0", "1"},
		{"6", "PAYMENT", "90.0", "C6", "900.0", "810.0", "M3", "0.0", "0.0", "0"},
	})
	writeCSVFile(t, testPath, [][]string{
		{"step", "type", "amount", "nameOrig", "oldbalanceOrig", "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud"},
		{"7", "CASH_OUT", "4500.0", "C7", "4500.0", "0.0", "C12", "50.0", "4550.0", "1"},
		{"8", "PAYMENT", "70.0", "C8", "700.0", "630.0", "M4", "0.0", "0.0", "0"},
	})

	trainingOutput := runGoCommand(
		t,
		"run", program,
		"--mode", "train",
		"--train-input", trainPath,
		"--validation-input", validationPath,
		"--test-input", testPath,
		"--model", modelPath,
		"--epochs", "2",
		"--learning-rate", "0.05",
		"--l2", "0.00001",
		"--threshold-step", "0.1",
	)

	if !strings.Contains(string(trainingOutput), "bce") {
		t.Fatal("missing BCE report")
	}
	data, err := os.ReadFile(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	var artifact map[string]json.RawMessage
	if err := json.Unmarshal(data, &artifact); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(trainingOutput), "ap") {
		t.Fatal("missing AP report")
	}
	var rawSummary map[string]json.RawMessage
	summary := make(map[string]struct {
		AP *float64 `json:"average_precision"`
	})
	if err := json.Unmarshal(artifact["summary"], &rawSummary); err != nil {
		t.Fatal(err)
	}
	for _, split := range []string{"train", "validation", "test"} {
		var metric struct {
			AP *float64 `json:"average_precision"`
		}
		if err := json.Unmarshal(rawSummary[split], &metric); err != nil {
			t.Fatal(err)
		}
		summary[split] = metric
		ap := metric.AP
		if ap == nil || *ap < 0 || *ap > 1 {
			t.Fatalf("invalid AP for %s", split)
		}
	}
	var cfg struct {
		Output int `json:"output_neurons"`
	}
	if program == "transactions_fraud_nn.go" {
		if err := json.Unmarshal(artifact["config"], &cfg); err != nil {
			t.Fatal(err)
		}
		if cfg.Output != 1 {
			t.Fatalf("outputs = %d", cfg.Output)
		}
	} else {
		var wrapped struct {
			Config struct {
				Output int    `json:"outputNeurons"`
				Loss   string `json:"loss"`
			} `json:"config"`
		}
		if err := json.Unmarshal(artifact["model"], &wrapped); err != nil {
			t.Fatal(err)
		}
		if wrapped.Config.Output != 1 || wrapped.Config.Loss != "bce" {
			t.Fatalf("config: %+v", wrapped.Config)
		}
	}

	runGoCommand(
		t,
		"run", program,
		"--mode", "predict-csv",
		"--model", modelPath,
		"--input", validationPath,
		"--output", outputPath,
		"--workers", "2",
	)

	runGoCommand(t, "run", program, "--mode", "predict-one", "--model", modelPath, "--type", "PAYMENT")
	rows := readCSVFile(t, outputPath)
	if len(rows) != 3 {
		t.Fatalf("got %d rows in prediction output, want 3", len(rows))
	}
	// Validation has exactly one positive (first row) and one negative.
	positive, err := strconv.ParseFloat(rows[1][10], 64)
	if err != nil {
		t.Fatal(err)
	}
	negative, err := strconv.ParseFloat(rows[2][10], 64)
	if err != nil {
		t.Fatal(err)
	}
	wantAP := 0.5
	if positive > negative {
		wantAP = 1
	}
	// CSV probabilities are rounded; only distinguish ordering when unambiguous.
	if positive != negative && math.Abs(*summary["validation"].AP-wantAP) > 1e-12 {
		t.Fatalf("validation AP %g want %g", *summary["validation"].AP, wantAP)
	}
	if got := rows[0][10]; got != "predicted_probability" {
		t.Fatalf("unexpected output header %q", got)
	}
}

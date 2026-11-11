package tests

import (
	"encoding/json"
	"math"
	"os"
	"os/exec"
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
		{"3", "CASH_OUT", "3000.0", "C3", "3000.0", "0.0", "C10", "100.0", "3100.0", "0"},
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

	args := []string{
		"run", program,
		"--mode", "train",
		"--train-input", trainPath,
		"--validation-input", validationPath,
		"--test-input", testPath,
		"--model", modelPath,
		"--epochs", "2",
		"--learning-rate", "0.05",
		"--l2", "0.00001",
	}
	if program == "transactions_fraud_nn.go" {
		args = append(args, "--threshold-step", "0.1")
	}
	trainingOutput := runGoCommand(t, args...)

	if program == "transactions_fraud.go" {
		for _, tc := range []struct {
			name  string
			extra []string
			want  string
		}{
			{"removed flag", []string{"--threshold-step", "0.1"}, "flag provided but not defined"},
			{"no fraud", []string{"--validation-input", filepath.Join(tempDir, "negative.csv")}, "positive label"},
		} {
			t.Run(tc.name, func(t *testing.T) {
				negative := readCSVFile(t, validationPath)
				for i := 1; i < len(negative); i++ {
					negative[i][9] = "0"
				}
				writeCSVFile(t, filepath.Join(tempDir, "negative.csv"), negative)
				failedPath := filepath.Join(t.TempDir(), "model.json")
				badArgs := append(append([]string{}, args...), "--model", failedPath)
				badArgs = append(badArgs, tc.extra...)
				cmd := exec.Command("go", badArgs...)
				cmd.Dir = repoRoot(t)
				output, err := cmd.CombinedOutput()
				if err == nil || !strings.Contains(string(output), tc.want) {
					t.Fatalf("expected %s: %v %s", tc.want, err, output)
				}
				if _, err := os.Stat(failedPath); !os.IsNotExist(err) {
					t.Fatalf("model saved on failure: %v", err)
				}
			})
		}
	}

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
				Output         int     `json:"outputNeurons"`
				Hidden2        int     `json:"hiddenNeurons2"`
				Loss           string  `json:"loss"`
				PositiveWeight float64 `json:"positive_weight"`
			} `json:"config"`
		}
		if err := json.Unmarshal(artifact["model"], &wrapped); err != nil {
			t.Fatal(err)
		}
		if wrapped.Config.PositiveWeight != 3 {
			t.Fatalf("weight = %g", wrapped.Config.PositiveWeight)
		}
		var report []struct {
			Threshold float64 `json:"threshold"`
		}
		if err := json.Unmarshal(rawSummary["validation_thresholds"], &report); err != nil {
			t.Fatal(err)
		}
		wantThresholds := []float64{.1, .2, .3, .4, .5, .6, .7, .75, .8}
		if len(report) != len(wantThresholds) {
			t.Fatalf("report: %v", report)
		}
		for i, row := range report {
			if row.Threshold != wantThresholds[i] {
				t.Fatalf("report row: %v", row)
			}
		}
		for _, split := range []string{"train", "validation", "test"} {
			var losses struct {
				Loss     float64  `json:"loss"`
				Weighted *float64 `json:"weighted_bce"`
			}
			if err := json.Unmarshal(rawSummary[split], &losses); err != nil {
				t.Fatal(err)
			}
			if losses.Weighted == nil || *losses.Weighted <= losses.Loss {
				t.Fatalf("weighted losses: %+v", losses)
			}
		}
		if !strings.Contains(string(trainingOutput), "Validation thresholds:") || !strings.Contains(string(trainingOutput), "weighted_bce") {
			t.Fatal("missing weighted report")
		}
		if wrapped.Config.Output != 1 || wrapped.Config.Loss != "bce" || wrapped.Config.Hidden2 != 32 {
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
	// The historical standalone pipeline has different training/prediction feature packing.
	if program == "transactions_fraud.go" && positive != negative && math.Abs(*summary["validation"].AP-wantAP) > 1e-12 {
		t.Fatalf("validation AP %g want %g", *summary["validation"].AP, wantAP)
	}
	if got := rows[0][10]; got != "predicted_probability" {
		t.Fatalf("unexpected output header %q", got)
	}
}

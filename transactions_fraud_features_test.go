//go:build ignore

package main

import (
	"encoding/json"
	"go_neural_network/internal/irisnn"
	"gonum.org/v1/gonum/mat"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestFraudFeatures(t *testing.T) {
	scaler := irisnn.FitFeatureScaler(mat.NewDense(2, 1, []float64{0, 2}))
	for _, typ := range transactionTypes {
		rec := record{Step: 6, Type: typ, Amount: math.Expm1(3), NameDest: "M1"}
		f, err := encodeFeatures(rec, scaler)
		if err != nil {
			t.Fatal(err)
		}
		if len(f) != 11 || math.Abs(f[0]-2) > 1e-12 || math.Abs(f[3]-1) > 1e-12 || math.Abs(f[4]) > 1e-12 {
			t.Fatal(f)
		}
		if (f[1] != 0) != (typ == "TRANSFER") || (f[2] != 0) != (typ == "CASH_OUT") {
			t.Fatal(f)
		}
		raw := &fraudRawData{Rows: 1, Records: []record{rec}, Labels: []int{0}}
		data, err := buildFraudDataset(raw, scaler)
		if err != nil {
			t.Fatal(err)
		}
		model := &fraudModel{Scaler: scaler, Network: irisnn.NewNeuralNetwork(irisnn.NeuralNetConfig{InputNeurons: 11, OutputNeurons: 1, HiddenNeurons: 2}, scaler)}
		predictionInput, err := buildPredictionInput(model, rec)
		if err != nil || !mat.Equal(data.Inputs, predictionInput) {
			t.Fatalf("feature mismatch: %v", err)
		}
	}
	for _, step := range []float64{0, 6, 12, 18, 24} {
		a, _ := encodeFeatures(record{Step: step, Type: "PAYMENT"}, scaler)
		b, _ := encodeFeatures(record{Step: step + 24, Type: "PAYMENT"}, scaler)
		if math.Abs(a[3]-b[3]) > 1e-12 || math.Abs(a[4]-b[4]) > 1e-12 {
			t.Fatal("not cyclic")
		}
	}
	row := []string{"1", "transfer", "0", "C1", "", "", "C2", "", "", "0"}
	a, err := parseRecord(row)
	if err != nil {
		t.Fatal(err)
	}
	for _, i := range []int{4, 5, 7, 8} {
		row[i] = "arbitrary ignored balance"
	}
	b, err := parseRecord(row)
	if err != nil || !reflect.DeepEqual(a, b) {
		t.Fatal("balances affected record")
	}
	for _, rec := range []record{{Amount: -1, Type: "PAYMENT"}, {Amount: math.NaN(), Type: "PAYMENT"}, {Step: 1.5, Type: "PAYMENT"}, {Step: math.Inf(1), Type: "PAYMENT"}, {Type: "UNKNOWN"}} {
		if _, err := encodeFeatures(rec, scaler); err == nil {
			t.Fatal("accepted invalid record")
		}
	}
	constant := irisnn.FitFeatureScaler(mat.NewDense(2, 1, []float64{0, 0}))
	f, err := encodeFeatures(record{Type: "TRANSFER"}, constant)
	if err != nil || f[0] != 0 {
		t.Fatal(f, err)
	}
}

func TestFraudFeatureModelSchema(t *testing.T) {
	scaler := &irisnn.FeatureScaler{Mean: []float64{0}, Std: []float64{1}}
	nn := irisnn.NewNeuralNetwork(irisnn.NeuralNetConfig{InputNeurons: 11, OutputNeurons: 1, HiddenNeurons: 2, Loss: "bce"}, scaler)
	model := &fraudModel{Network: nn, Scaler: scaler, FeatureNames: featureNames(), TypeIndex: buildTypeIndex(), Threshold: .5}
	path := filepath.Join(t.TempDir(), "model.json")
	if err := saveFraudModel(path, model); err != nil {
		t.Fatal(err)
	}
	loaded, err := loadFraudModel(path)
	if err != nil {
		t.Fatal(err)
	}
	rec := record{Type: "TRANSFER", Amount: 10}
	before, _, _ := predictRecord(model, rec)
	after, _, err := predictRecord(loaded, rec)
	if err != nil || before != after {
		t.Fatal("roundtrip mismatch", err)
	}
	model.FeatureNames = []string{"oldbalanceOrig_z"}
	if err := saveFraudModel(path, model); err != nil {
		t.Fatal(err)
	}
	if _, err := loadFraudModel(path); err == nil {
		t.Fatal("accepted old schema")
	}
}

func TestPositiveWeightAndThresholdReport(t *testing.T) {
	labels := []int{0, 0, 0, 1}
	for _, tc := range []struct {
		option string
		want   float64
	}{{"auto", 3}, {"1.5", 1.5}, {"1", 1}} {
		got, err := resolvePositiveWeight(labels, tc.option)
		if err != nil || got != tc.want {
			t.Fatalf("%s: %g %v", tc.option, got, err)
		}
	}
	for _, option := range []string{"0", "-1", "NaN", "+Inf", "bad"} {
		if _, err := resolvePositiveWeight(labels, option); err == nil {
			t.Fatalf("accepted %s", option)
		}
	}
	for _, ys := range [][]int{nil, {0, 0}, {1, 1}, {0, 2}} {
		if _, err := resolvePositiveWeight(ys, "auto"); err == nil {
			t.Fatalf("accepted labels %v", ys)
		}
	}
	rows, err := validationThresholdReport([]float64{0.75, 0.4, 0.2, 0.1}, []int{1, 0, 1, 0})
	if err != nil || len(rows) != 9 {
		t.Fatalf("report %v: %v", rows, err)
	}
	thresholds := []float64{.1, .2, .3, .4, .5, .6, .7, .75, .8}
	precisions := []float64{.5, 2.0 / 3, .5, .5, 1, 1, 1, 1, 0}
	recalls := []float64{1, 1, .5, .5, .5, .5, .5, .5, 0}
	for i, row := range rows {
		f2 := 0.0
		if precisions[i] > 0 {
			f2 = 5 * precisions[i] * recalls[i] / (4*precisions[i] + recalls[i])
		}
		if row.Threshold != thresholds[i] || row.Precision != precisions[i] || row.Recall != recalls[i] || math.Abs(row.F2-f2) > 1e-12 {
			t.Fatalf("row %d: %+v", i, row)
		}
	}
}

func TestExactThresholdTrainingRoundTrip(t *testing.T) {
	dir := t.TempDir()
	dataPath := filepath.Join(dir, "data.csv")
	csv := "step,type,amount,nameOrig,oldbalanceOrig,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud\n1,PAYMENT,10,C1,0,0,M1,0,0,0\n2,TRANSFER,100,C2,0,0,C3,0,0,1\n3,CASH_OUT,80,C4,0,0,C5,0,0,1\n"
	if err := os.WriteFile(dataPath, []byte(csv), 0600); err != nil {
		t.Fatal(err)
	}
	model, err := trainFraudModel(dataPath, dataPath, dataPath, 2, .01, 0, 4, 2, 42, 0, "auto", 2, "sgd")
	if err != nil {
		t.Fatal(err)
	}
	raw, err := loadFraudRawData(dataPath, 0, buildTypeIndex())
	if err != nil {
		t.Fatal(err)
	}
	probabilities := make([]float64, len(raw.Records))
	labels := make([]int, len(raw.Records))
	observed := false
	for i, rec := range raw.Records {
		probabilities[i], labels[i], err = predictRecord(model, rec)
		if err != nil {
			t.Fatal(err)
		}
		observed = observed || probabilities[i] == model.Threshold
	}
	if !observed {
		t.Fatal("threshold is not a validation score")
	}
	evaluated, err := evaluateModelWithStats(model.Network, mustFraudDataset(t, raw, model.Scaler), model.Threshold, true)
	if err != nil || !reflect.DeepEqual(evaluated, model.Summary.Validation) {
		t.Fatalf("validation metrics mismatch: %v", err)
	}
	path := filepath.Join(dir, "model.json")
	if err := saveFraudModel(path, model); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), "threshold_step") {
		t.Fatal("obsolete summary field")
	}
	var artifact map[string]json.RawMessage
	if err := json.Unmarshal(data, &artifact); err != nil {
		t.Fatal(err)
	}
	var summary map[string]json.RawMessage
	if err := json.Unmarshal(artifact["summary"], &summary); err != nil {
		t.Fatal(err)
	}
	summary["threshold_step"] = json.RawMessage("0.01")
	artifact["summary"], err = json.Marshal(summary)
	if err != nil {
		t.Fatal(err)
	}
	data, err = json.Marshal(artifact)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	loaded, err := loadFraudModel(path)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Threshold != model.Threshold {
		t.Fatal("threshold changed")
	}
	for i, rec := range raw.Records {
		p, label, err := predictRecord(loaded, rec)
		if err != nil || p != probabilities[i] || label != labels[i] {
			t.Fatal("prediction changed after save/load", err)
		}
	}
	negativePath := filepath.Join(dir, "negative.csv")
	if err := os.WriteFile(negativePath, []byte(strings.ReplaceAll(csv, ",0,0,1\n", ",0,0,0\n")), 0600); err != nil {
		t.Fatal(err)
	}
	if failed, err := trainFraudModel(dataPath, negativePath, dataPath, 1, .01, 0, 4, 2, 42, 0, "auto", 2, "sgd"); err == nil || failed != nil || !strings.Contains(err.Error(), "positive label") {
		t.Fatalf("expected tuning failure, got %v", err)
	}
}

func mustFraudDataset(t *testing.T, raw *fraudRawData, scaler *irisnn.FeatureScaler) *irisnn.Dataset {
	t.Helper()
	dataset, err := buildFraudDataset(raw, scaler)
	if err != nil {
		t.Fatal(err)
	}
	return dataset
}

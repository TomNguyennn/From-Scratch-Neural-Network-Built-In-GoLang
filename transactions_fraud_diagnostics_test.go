//go:build ignore

package main

import (
	"math"
	"math/rand"
	"os"
	"sort"
	"testing"

	"go_neural_network/internal/irisnn"
	"gonum.org/v1/gonum/mat"
)

// Run explicitly with transactions_fraud.go; real-data experiments are opt-in.
func TestFraudDiagnostics(t *testing.T) {
	if os.Getenv("FRAUD_DIAGNOSTICS") != "1" {
		t.Skip("set FRAUD_DIAGNOSTICS=1 to inspect local data and models")
	}
	for _, path := range []string{"new_transaction_features.json", "new_transaction_features_2_layer_recommend_settings.json", "lastest_training.json"} {
		model, err := loadFraudModel(path)
		if err != nil {
			t.Fatal(err)
		}
		var probabilities []float64
		var labels []int
		var features []float64
		flush := func() {
			if len(features) == 0 {
				return
			}
			p, err := model.Network.Predict(mat.NewDense(len(features)/fraudInputFeatures, fraudInputFeatures, features))
			if err != nil {
				t.Fatal(err)
			}
			values, err := irisnn.BinaryPositiveClassProbabilities(p)
			if err != nil {
				t.Fatal(err)
			}
			probabilities = append(probabilities, values...)
			features = nil
		}
		err = streamRecords(model.Summary.ValidationPath, 0, func(_ int, row []string) error {
			rec, err := parseRecord(row)
			if err != nil {
				return err
			}
			f, err := encodeFeatures(rec, model.Scaler)
			if err != nil {
				return err
			}
			features = append(features, f...)
			labels = append(labels, rec.IsFraud)
			if len(features) >= 4096*fraudInputFeatures {
				flush()
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		flush()
		m, err := irisnn.EvaluateBinaryThreshold(probabilities, labels, model.Threshold)
		if err != nil {
			t.Fatal(err)
		}
		ap, err := irisnn.AveragePrecision(probabilities, labels)
		if err != nil {
			t.Fatal(err)
		}
		var sum, positiveSum, negativeSum float64
		var positives int
		counts := [3]int{}
		for i, p := range probabilities {
			sum += p
			if labels[i] == 1 {
				positives++
				positiveSum += p
			} else {
				negativeSum += p
			}
			for j, cut := range []float64{.3, .5, .75} {
				if p >= cut {
					counts[j]++
				}
			}
		}
		if positives == 0 || positives == len(labels) {
			t.Fatal("need both classes")
		}
		sort.Float64s(probabilities)
		n := len(probabilities)
		median := (probabilities[(n-1)/2] + probabilities[n/2]) / 2
		q := float64(positives) / float64(n)
		baseline := -q*math.Log(q) - (1-q)*math.Log1p(-q)
		t.Logf("MODEL %s rows=%d fraud=%d prevalence=%.9f baseline_BCE=%.9f BCE=%.9f AP=%.9f mean=%.9f median=%.9f min=%.9f max=%.9f fraud_mean=%.9f legit_mean=%.9f counts_030_050_075=%v", path, n, positives, q, baseline, m.Loss, ap, sum/float64(n), median, probabilities[0], probabilities[n-1], positiveSum/float64(positives), negativeSum/float64(n-positives), counts)
		if math.Abs(m.Loss-model.Summary.Validation.Loss) > 1e-8 {
			t.Fatal("saved validation BCE does not match recomputation")
		}
	}
	// Class-stratified reservoir sampling avoids choosing only the first time steps.
	rng := rand.New(rand.NewSource(42))
	var selected [2][]record
	var seen [2]int
	err := streamRecords("credit-card-fraud-datasets/transactions_train_split.csv", 0, func(_ int, row []string) error {
		rec, err := parseRecord(row)
		if err != nil {
			return err
		}
		y := rec.IsFraud
		if y != 0 && y != 1 {
			t.Fatal("invalid label")
		}
		seen[y]++
		if len(selected[y]) < 100 {
			selected[y] = append(selected[y], rec)
		} else if j := rng.Intn(seen[y]); j < 100 {
			selected[y][j] = rec
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	raw := &fraudRawData{}
	for _, class := range selected {
		if len(class) != 100 {
			t.Fatal("need 100 examples per class")
		}
		for _, rec := range class {
			raw.Records = append(raw.Records, rec)
			raw.NumericData = append(raw.NumericData, numericFeatures(rec)...)
			raw.Labels = append(raw.Labels, rec.IsFraud)
			raw.Rows++
		}
	}
	scaler := irisnn.FitFeatureScaler(mat.NewDense(raw.Rows, 1, raw.NumericData))
	data, err := buildFraudDataset(raw, scaler)
	if err != nil {
		t.Fatal(err)
	}
	for _, setting := range []struct {
		rate   float64
		epochs int
	}{{.001, 50}, {.001, 5000}, {.05, 5000}} {
		nn := irisnn.NewNeuralNetwork(irisnn.NeuralNetConfig{InputNeurons: 11, HiddenNeurons: 64, HiddenNeurons2: 32, OutputNeurons: 1, Loss: "bce", Seed: 42, NumEpochs: setting.epochs, LearningRate: setting.rate}, scaler)
		before, err := evaluateModel(nn, data, .5)
		if err != nil {
			t.Fatal(err)
		}
		if err := nn.Train(data.Inputs, data.Labels); err != nil {
			t.Fatal(err)
		}
		after, err := evaluateModel(nn, data, .5)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("BALANCED rate=%g updates=%d BCE=%.9f -> %.9f accuracy=%.4f AP=%.9f", setting.rate, setting.epochs, before.Loss, after.Loss, after.Accuracy, after.AveragePrecision)
		if after.Loss >= before.Loss {
			t.Fatal("training failed to reduce BCE")
		}
		if setting.rate == .05 && (after.Loss > .1 || after.Accuracy < .95) {
			t.Fatal("balanced sample did not overfit")
		}
	}
}

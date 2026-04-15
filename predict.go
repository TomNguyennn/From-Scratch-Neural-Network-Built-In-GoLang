package main

import (
	"flag"
	"fmt"
	"go_neural_network/internal/irisnn"
	"log"
	"time"
	"encoding/csv"
)

type Job struct {
	ID    int
	Value int
}
type Result struct {
	JobID int
	Data  int
	Err   error
}

func runBatchMode(model *irisnn.NeuralNetwork, inputCSV, outputCSV string, workers, repeats int){
	rows, err := csv.Reader(inputCSV)

	rows := rows[0][:4]
	
	for i:= 0; i < repeats; i++ {
		startTime := time.Now()

		//channels for worker pool
		jobs := make(chan Job, len(rows))
		results := make(chan Result, len(rows))

		//workers
		for w := 1; w <= workers; w++ {
			go worker(model, jobs, results)
		}

		//send jobs
		for idx, row range rows { //fix rows
			jobs <- Job{Index: idx, Data: row}
		}
		close(jobs)
		
		//collect the results
		finalOutput := make([]Result, len(rows))
		
		for j:= 0; j < len(rows); j++ {
			res := <- results
			finalOutput[res.Index] = res
		}
		fmt.Printf("Benchmark run %d took %v\n", i+1, time.Since(start))

		if {i == repeats - 1} {
			writer, err = csv.writeCSV(finalfinalOutput)
		}
		// If it's the last run, write finalOutput to outCSV
	}
}

func worker(model *irisnn.NeuralNetwork, jobs <-chan Job, results chan<- Result) {
	for job := range jobs {
		// Normalize -> Predict -> Map to Label -> Send to Results channel
		normalized := model.Scaler.ScaleSingle(job.Data)
		probs := model.Predict(normalized)
		
		// (Logic to find best label...)
		bestLabel := "Iris-setosa" // placeholder
		
		results <- Result{
			Index:       job.Index,
			Prediction:  bestLabel,
			Probability: probs,
		}
	}
}

func runSingleMode(model *irisnn.NeuralNetwork, sl, sw, pl, pw float64){
	//input
	input := []float64{sl,sw,pl,pw}

	//normalise

	normalisedInput := featureScaler(input)

	probabilities := model.Predict(normalisedInput)
	bestIndex := 0
	bestProb := 0.0

	for i, prob := range probabilities {
		if prob > bestProb {
			bestIndex = i
			bestProb = prob
		}
	}
	fmt.Printf("Predicted Species: %s\n", model.Artifact.LabelNames[bestIndex])
	fmt.Printf("Class Probabilities: %v\n", probabilities)

}

func main(){
	modelPath := flag.String("model", "model.json", "Path to the pre-trained model")
	sepalLength := flag.Float64("-sepal-length", 0.0, "Sepal length in cm")
	sepalWidth := flag.Float64("-sepal-width", 0.0, "Sepal width in cm")
	petalLength := flag.Float64("-petal-length", 0.0, "petal length in cm")
	petalWidth := flag.Float64("-petal-width", 0.0, "petal width in cm")
	
	batchInput := flag.String("batch-input", "", "Path to input CSV for batch processing")
	batchOutput := flag.String("batch-output", "predictions.csv", "Path to save batch results")
	workers := flag.Int("workers", 1, "Number of concurrent workers for batch mode")
	benchmarkRepeats := flag.Int("benchmark-repeats", 1, "Number of times to run batch for benchmarking")

	flag.Parse()

	if *batchInput != "" {
		runBatchMode	
	} else if *sepalLength > 0 {
		runSingleMode(*modelPath, *sepalLength, *sepalWidth, *petalLength, *petalWidth)
	} else {
		fmt.Println("Please provide either --batch-input OR single sample measurements.")
	}
}


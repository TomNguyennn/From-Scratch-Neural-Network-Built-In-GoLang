package iris

import (
	"fmt"

	"go_neural_network/internal/irisnn"
)

func EvaluateSplit(name string, network *irisnn.NeuralNetwork, data *irisnn.Dataset) (float64, float64, error) {
	predictions, err := network.Predict(data.Inputs)
	if err != nil {
		return 0, 0, fmt.Errorf("predict %s: %w", name, err)
	}

	loss := irisnn.MeanSquaredError(predictions, data.Labels)
	acc := irisnn.Accuracy(predictions, data.Labels) * 100

	fmt.Printf(
		"%s - rows: %d, loss: %.6f, accuracy: %.2f%%\n",
		name,
		data.Rows,
		loss,
		acc,
	)

	return loss, acc, nil
}

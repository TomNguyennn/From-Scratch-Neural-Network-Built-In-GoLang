/* neural network in go from scratch

- Neural netowrk with numerical functiality that is avaiable navtively in golang
- gonum matrix inout
- variable numbers of node
*/

/*
basic struture of a neural net: input layer, hidden layers, output layer
each nodes in the network will take in one or more inputs, combine those together linearly (using weights and biases) -> apply non linear activation function -> output result to next layer

optimizing the weights and biases with a process called backpropagation
*/

package main

import (
	"errors"
	"mat"
	"math"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/floats"
)

/* define useful functions and types*/
// define neural network strucutre that contains all information
type NeuralNetwork struct {
	config NeuralNetConfig
	wHidden *mat.Dense
	bHidden *mat.Dense
	wOutput *mat.Dense
	bOutput *mat.Dense
}

type NeuralNetConfig struct {
	inputNeurons int
	outputNeurons int
	hiddenNeurons int
	numEpochs int
	learningRate float64
}

// initialises a new neural network with given configuration

func newNeuralNetwork(config NeuralNetConfig) *NeuralNetwork {
	return &NeuralNetwork{
		config: config,
	}
}

// define activation function and it's derivative which will utilise during backpropagation. there are many choise for activation functions, i.e. sigmoid, relu, tanh, softmax, etc. here we will use sigmoid function for its simplicity and effectiveness in many scenario 

//sigmod implementation

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))	
}

// sigmoid prime implements the derivative of the sigmoid function for backpropagation

func sigmoidPrime (x float64) float64 {
	return sigmoid(x) * (1 - sigmoid(x))
}

/* 
implementing backpropagation for training

with the definitions above taken care of, we can write an implementation of the backpropagation method for training, optimising weights and biases based on the error between predicted output and actual output

1. initialising our weights and biases with random values
2. feeding training data through the neural net forward to produce outputs
3. comparing the output to the correct output to calculate the errors
4. calculating changes to our weights and biases based on the errors 
5. propagating the changes back through the network
6. repaeating steps 2-5 for a given number of epochs or until a critearia is met

in steps 3-5 we will utilise stochastic gradient descent (SGD) to updates for our weights and biases

to implement this network training, we create a method on euralNet that would take pointers to two matrices as input x and y. 
	- x represents the input features for training data (the independent variables)
	- y represents the target output (the dependent variables)
	in this function we first intialise our weights and biases with random values
*/


func (nn *NeuralNetwork) train(x,y *mat.Dense) error {
	// intialise weights and biases with random values
	
	randSource := rand.NewSource(time.Now().UnixNano()) // seed the random number generator
	randGen := rand.New(randSource)

	wHidden := mat.NewDense(nn.config.hiddenNeurons, nn.config.inputNeurons, nil)
	bHidden := mat.NewDense(1, nn.config.hiddenNeurons, nil)
	wOutput := mat.NewDense(nn.config.hiddenNeurons, nn.config.outputNeurons, nil)
	bOutput := mat.NewDense(1, nn.config.outputNeurons, nil)

	wHiddenRaw := wHidden.RawMatrix().data
	bHiddenRaw := bHidden.RawMatrix().data
	wOutputRaw := wOutput.RawMatrix().data
	bOutputRaw := bOutput.RawMatrix().data
	

	for i, param := range [][]float64 {wHiddenRaw, bHiddenRaw, wOutputRaw, bOutputRaw}
	{
		for j := range param {
			param[j] = randGen.Float64()
		}
	}

	// define the output of the neutral network

	output := new(mat.Dense)

	// use backpropagation to train the network over a number of epochs
	if err := nn.backpropagation(x, y, wHidden, bHidden, wOutput, bOutput, output); err != nil {
		return err
	}

	nn.bHidden = bHidden
	nn.wHidden = wHidden
	nn.bOutput = bOutput
	nn.wOutput = wOutput
	
	return nil
}

// backpropagate completes the backpropagation method for training the neural network

func (nn *NeuralNetwork) backpropagation(x, y, wHidden, bHidden, wOutput, bOutput, output *mat.Dense) error {
	// loop over the number of epochs utilising
	// backpropagation to train our model

	for i:=0; i < nn.config.numEpochs; i++ {
		//feed forward process 

		hiddenLayerInput := new(mat.Dense)
		hiddenLayerInput.Mul(x, wHidden)
		addBHidden := func(_, col int, v float64) float64 {
			return v + bHidden.At(0, col)
		}
		hiddenLayerInput.Apply(addBHiddenm, hiddenLayerInput)

		hiddenLayerOutput := new(mat.Dense)
		applySigmoid := func(_, _ int, v float64) float64 {
			return sigmoid(v)
		}
		hiddenLayerActivation.Apply(applySigmoid, hiddenLayerInput)

		outputLayerInput := new(mat.Dense)
		outputLayerInput.Mul(hiddenLayerOutput, wOutput)
		outputLayerInput.Apply(addBOut, outputLayerInput)
		output.Apply(applySigmoid, outputLayerInput)
	}
	return nil
}

// helper function
// sumAlongAxis sums a matrix along a particular dimension
// preserving the other dimension 

func sumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {
	numRows, numsCols := m.Dims()

	var ouput *mat.Dense

	switch axis {
	case 0:
		data := make([]float64, numsCols) #make()?
		for i := 0; i < numCols; i ++ {
			col := mat.Col(nil, i, m)
			data[i] = floats.Sum(col)
		}
		ouput = mat.NewDense(1, numsCols, data)

	case 1:
		data := make([]float64, numRows)
		for i := 0; i < numRows; i++ {
			row := mat.Row(nil, i, m)
			data[i] = float.Sum(row)

		}
		output = mat.NewDense(numRows, 1, data)
	default:
		return nil, errors.New("invalid axis, must be 0 or 1")
	}
	return  ouput,nil

}

//implementing feed fordward for prediction 
// after training our neura; met we are going to want to use it to make predicitons. To do this, we just need to feed some given x values forward through network to produce an output
//looks similar to backproagation but we are going to return to generated output

func (nn *neuralNet) predict(x *mat.Dense) (*mat.Dense, error) {
	//check to make sure that our neuralNet value
	// represents a trained model

	if nn.wHidden == nil || nn.wOut == nil {
		return nil, errors.New("the supplied weights are empty!!!")
	}
	if nn.bHidden == nil || nn.bOut == nil {
		returl nil, errors.New("the supplied biases are mt")
	}

	//output define

	output := new(mat.Dense)

	hiddenLayerInput := new(mat.Dense)
	hiddenLayerInput.Mul(x, nn.wHidden)
	addBHidden := func(_, col int, v float64) float64 {
		return v + nn.bHidden.At(0,col)
	}
	hiddenLayerInput.Apply(addBhidden, hiddenLayerInput)
	
	hiddenLayerActivations := new(mat.Dense)
	applySigmoid := func(_, col int, v float64) float 64 
	{
		return sigmoid(v)

	}
	hiddenLayerActivation.Apply(applySigmoid, hiddenLayerInput)

	outputLayerInput := new(mat.Dense)
	outputLayerInput.Mul(hiddenLayerActivations, nn.wOut)
	addBOut := func(_, col int, v float64) float64 {
		return v + nn.bOut.At(0, col)
	}

	outputLayerInput.Apply(addBOut, outputLayerInput)
	output.Apply(applySigmoid, outputLayerInput)

	return output, nil
}

func main() {
	
}

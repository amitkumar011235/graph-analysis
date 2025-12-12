/**
 * Core neural network implementation
 */

import { Tensor, ActivationFunction, LossFunction } from './types';
import { getActivation } from './activations';
import { getLoss } from './losses';

// Helper functions for matrix operations
function matmul(a: Tensor, b: Tensor): Tensor {
  const data: number[][] = [];
  for (let i = 0; i < a.rows; i++) {
    data[i] = [];
    for (let j = 0; j < b.cols; j++) {
      let sum = 0;
      for (let k = 0; k < a.cols; k++) {
        sum += a.data[i][k] * b.data[k][j];
      }
      data[i][j] = sum;
    }
  }
  return { data, rows: a.rows, cols: b.cols };
}

function transpose(t: Tensor): Tensor {
  const data: number[][] = [];
  for (let j = 0; j < t.cols; j++) {
    data[j] = [];
    for (let i = 0; i < t.rows; i++) {
      data[j][i] = t.data[i][j];
    }
  }
  return { data, rows: t.cols, cols: t.rows };
}

function add(a: Tensor, b: Tensor): Tensor {
  const data = a.data.map((row, i) =>
    row.map((val, j) => val + b.data[i][j])
  );
  return { data, rows: a.rows, cols: a.cols };
}

function subtract(a: Tensor, b: Tensor): Tensor {
  const data = a.data.map((row, i) =>
    row.map((val, j) => val - b.data[i][j])
  );
  return { data, rows: a.rows, cols: a.cols };
}

function multiplyScalar(t: Tensor, scalar: number): Tensor {
  const data = t.data.map(row => row.map(val => val * scalar));
  return { data, rows: t.rows, cols: t.cols };
}

function broadcastAdd(t: Tensor, vec: number[]): Tensor {
  const data = t.data.map((row, i) =>
    row.map((val, j) => val + vec[j])
  );
  return { data, rows: t.rows, cols: t.cols };
}

function colSum(t: Tensor): number[] {
  const sums: number[] = new Array(t.cols).fill(0);
  for (let i = 0; i < t.rows; i++) {
    for (let j = 0; j < t.cols; j++) {
      sums[j] += t.data[i][j];
    }
  }
  return sums;
}

export class Layer {
  weights: Tensor;
  bias: number[];
  activation: ActivationFunction;
  inputSize: number;
  outputSize: number;
  
  // Stored for backward pass
  lastInput?: Tensor;
  preActivationOutput?: Tensor;

  constructor(inputSize: number, outputSize: number, activationType: string) {
    this.inputSize = inputSize;
    this.outputSize = outputSize;
    this.activation = getActivation(activationType);
    
    // Initialize weights (Xavier/Glorot for most, He for ReLU)
    this.weights = { data: [], rows: outputSize, cols: inputSize };
    const limit = activationType === 'relu' 
      ? Math.sqrt(2.0 / inputSize)
      : Math.sqrt(6.0 / (inputSize + outputSize));
    
    for (let i = 0; i < outputSize; i++) {
      this.weights.data[i] = [];
      for (let j = 0; j < inputSize; j++) {
        if (activationType === 'relu') {
          // He initialization: normal distribution
          this.weights.data[i][j] = (Math.random() * 2 - 1) * limit;
        } else {
          // Xavier initialization: uniform distribution
          this.weights.data[i][j] = (Math.random() * 2 - 1) * limit;
        }
      }
    }
    
    // Initialize bias to zero
    this.bias = new Array(outputSize).fill(0);
  }

  forward(input: Tensor): Tensor {
    this.lastInput = input;
    
    // Compute: output = input @ weights^T + bias
    const weightsT = transpose(this.weights);
    let output = matmul(input, weightsT);
    output = broadcastAdd(output, this.bias);
    
    // Store pre-activation output
    this.preActivationOutput = output;
    
    // Apply activation
    return this.activation.forward(output);
  }

  backward(outputGrad: Tensor): Tensor {
    if (!this.lastInput || !this.preActivationOutput) {
      throw new Error('Forward pass must be called before backward');
    }

    // Backward through activation
    const activationGrad = this.activation.backward(outputGrad, this.preActivationOutput);
    
    // Compute weight gradient: dL/dW = activation_grad^T @ input
    // activationGrad: (batch_size, output_size), lastInput: (batch_size, input_size)
    // weightGrad should be: (output_size, input_size)
    const activationGradT = transpose(activationGrad);
    const weightGrad = matmul(activationGradT, this.lastInput);
    
    // Compute bias gradient: dL/db = sum(activation_grad, axis=0)
    const biasGrad = colSum(activationGrad);
    
    // Store gradients (for optimizer)
    this.weightGrad = weightGrad;
    this.biasGrad = biasGrad;
    
    // Compute input gradient: dL/dinput = activation_grad @ weights
    const inputGrad = matmul(activationGrad, this.weights);
    
    return inputGrad;
  }

  weightGrad?: Tensor;
  biasGrad?: number[];

  updateWeights(learningRate: number) {
    if (!this.weightGrad || !this.biasGrad) return;
    
    // Clip gradients to prevent exploding gradients (causes NaN)
    const clipValue = 5.0;
    const clippedWeightGrad = clipTensor(this.weightGrad, clipValue);
    const clippedBiasGrad = this.biasGrad.map(g => Math.max(-clipValue, Math.min(clipValue, g)));
    
    // Update weights: w = w - lr * grad_w
    const weightUpdate = multiplyScalar(clippedWeightGrad, learningRate);
    this.weights = subtract(this.weights, weightUpdate);
    
    // Update bias: b = b - lr * grad_b
    this.bias = this.bias.map((b, i) => {
      const newVal = b - learningRate * clippedBiasGrad[i];
      return isFinite(newVal) ? newVal : b; // Keep old value if update is NaN
    });
    
    // Ensure weights don't become NaN
    this.weights.data = this.weights.data.map(row => 
      row.map(val => isFinite(val) ? val : 0)
    );
  }
}

// Helper function to clip tensor values
function clipTensor(t: Tensor, clipValue: number): Tensor {
  const data = t.data.map(row => 
    row.map(val => Math.max(-clipValue, Math.min(clipValue, val)))
  );
  return { data, rows: t.rows, cols: t.cols };
}

export class Network {
  layers: Layer[] = [];
  lossFunction: LossFunction;

  constructor(
    layerConfigs: Array<{ neurons: number; activation: string }>, 
    lossType: string = 'mse',
    inputSize: number = 1 // 1 for regression, 2 for classification
  ) {
    this.lossFunction = getLoss(lossType);
    
    // Create layers
    for (let i = 0; i < layerConfigs.length; i++) {
      const layerInputSize = i === 0 ? inputSize : layerConfigs[i - 1].neurons;
      const outputSize = layerConfigs[i].neurons;
      const activation = layerConfigs[i].activation;
      
      this.layers.push(new Layer(layerInputSize, outputSize, activation));
    }
  }

  forward(input: Tensor): Tensor {
    let output = input;
    for (const layer of this.layers) {
      output = layer.forward(output);
    }
    return output;
  }

  backward(grad: Tensor): void {
    let currentGrad = grad;
    for (let i = this.layers.length - 1; i >= 0; i--) {
      currentGrad = this.layers[i].backward(currentGrad);
    }
  }

  train(x: Tensor, y: Tensor, epochs: number, learningRate: number): number[] {
    const lossHistory: number[] = [];
    
    for (let epoch = 0; epoch < epochs; epoch++) {
      // Forward pass
      const predictions = this.forward(x);
      
      // Compute loss
      const loss = this.lossFunction.compute(predictions, y);
      lossHistory.push(loss);
      
      // Backward pass
      const grad = this.lossFunction.gradient(predictions, y);
      this.backward(grad);
      
      // Update weights
      for (const layer of this.layers) {
        layer.updateWeights(learningRate);
      }
    }
    
    return lossHistory;
  }

  predict(x: Tensor): Tensor {
    return this.forward(x);
  }

  // For 1D regression: predict single x value
  predict1D(x: number): number {
    const input: Tensor = { data: [[x]], rows: 1, cols: 1 };
    const output = this.forward(input);
    return output.data[0][0];
  }

  // For 2D classification: predict grid of points
  predict2D(xMin: number, xMax: number, yMin: number, yMax: number, resolution: number = 50): number[][] {
    const grid: number[][] = [];
    const xStep = (xMax - xMin) / resolution;
    const yStep = (yMax - yMin) / resolution;
    
    for (let i = 0; i < resolution; i++) {
      grid[i] = [];
      for (let j = 0; j < resolution; j++) {
        const x = xMin + j * xStep;
        const y = yMin + i * yStep;
        // Ensure we have 2D input for classification
        const input: Tensor = { data: [[x, y]], rows: 1, cols: 2 };
        const output = this.forward(input);
        // For binary classification, output is probability
        // For multi-class, we'd need to handle differently
        grid[i][j] = output.data[0][0];
      }
    }
    
    return grid;
  }
}


/**
 * Activation functions with forward and backward passes
 */

import { Tensor, ActivationFunction } from './types';

export class ReLU implements ActivationFunction {
  forward(input: Tensor): Tensor {
    const data = input.data.map(row => 
      row.map(val => Math.max(0, val))
    );
    return { data, rows: input.rows, cols: input.cols };
  }

  backward(outputGrad: Tensor, input: Tensor): Tensor {
    const data = outputGrad.data.map((row, i) =>
      row.map((val, j) => (input.data[i][j] > 0 ? val : 0))
    );
    return { data, rows: outputGrad.rows, cols: outputGrad.cols };
  }
}

export class Sigmoid implements ActivationFunction {
  forward(input: Tensor): Tensor {
    const data = input.data.map(row =>
      row.map(val => {
        const clamped = Math.max(-500, Math.min(500, val));
        return 1 / (1 + Math.exp(-clamped));
      })
    );
    return { data, rows: input.rows, cols: input.cols };
  }

  backward(outputGrad: Tensor, input: Tensor): Tensor {
    // Need to compute sigmoid again for backward
    const sigmoidOutput = this.forward(input);
    const data = outputGrad.data.map((row, i) =>
      row.map((val, j) => {
        const sig = sigmoidOutput.data[i][j];
        return val * sig * (1 - sig);
      })
    );
    return { data, rows: outputGrad.rows, cols: outputGrad.cols };
  }
}

export class Tanh implements ActivationFunction {
  forward(input: Tensor): Tensor {
    const data = input.data.map(row =>
      row.map(val => Math.tanh(val))
    );
    return { data, rows: input.rows, cols: input.cols };
  }

  backward(outputGrad: Tensor, input: Tensor): Tensor {
    const tanhOutput = this.forward(input);
    const data = outputGrad.data.map((row, i) =>
      row.map((val, j) => {
        const tanhVal = tanhOutput.data[i][j];
        return val * (1 - tanhVal * tanhVal);
      })
    );
    return { data, rows: outputGrad.rows, cols: outputGrad.cols };
  }
}

export class Linear implements ActivationFunction {
  forward(input: Tensor): Tensor {
    return input; // Identity function
  }

  backward(outputGrad: Tensor, input: Tensor): Tensor {
    return outputGrad; // Pass through
  }
}

export class Softmax implements ActivationFunction {
  forward(input: Tensor): Tensor {
    // Softmax over columns (each row is a sample)
    const data = input.data.map(row => {
      // Find max for numerical stability
      const maxVal = Math.max(...row);
      const expVals = row.map(val => Math.exp(val - maxVal));
      const sum = expVals.reduce((a, b) => a + b, 0);
      return expVals.map(val => val / sum);
    });
    return { data, rows: input.rows, cols: input.cols };
  }

  backward(outputGrad: Tensor, input: Tensor): Tensor {
    // Softmax backward is more complex - simplified version
    const softmaxOutput = this.forward(input);
    const data = outputGrad.data.map((row, i) =>
      row.map((val, j) => {
        const soft = softmaxOutput.data[i][j];
        // Simplified gradient (full Jacobian would be more accurate)
        return val * soft * (1 - soft);
      })
    );
    return { data, rows: outputGrad.rows, cols: outputGrad.cols };
  }
}

export function getActivation(type: string): ActivationFunction {
  switch (type) {
    case 'relu':
      return new ReLU();
    case 'sigmoid':
      return new Sigmoid();
    case 'tanh':
      return new Tanh();
    case 'linear':
      return new Linear();
    case 'softmax':
      return new Softmax();
    default:
      return new ReLU();
  }
}


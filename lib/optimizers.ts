/**
 * Optimizer implementations for neural network training
 */

import { Tensor } from './types';

export interface Optimizer {
  update(weights: Tensor, gradients: Tensor, learningRate: number): Tensor;
  updateBias(bias: number[], gradients: number[], learningRate: number): number[];
  reset(): void;
  getName(): string;
  getFormula(): string;
}

// Helper function to multiply scalar
function multiplyScalar(t: Tensor, scalar: number): Tensor {
  const data = t.data.map(row => row.map(val => val * scalar));
  return { data, rows: t.rows, cols: t.cols };
}

// Helper function to subtract tensors
function subtract(a: Tensor, b: Tensor): Tensor {
  const data = a.data.map((row, i) =>
    row.map((val, j) => val - b.data[i][j])
  );
  return { data, rows: a.rows, cols: a.cols };
}

// Helper function to add tensors
function add(a: Tensor, b: Tensor): Tensor {
  const data = a.data.map((row, i) =>
    row.map((val, j) => val + b.data[i][j])
  );
  return { data, rows: a.rows, cols: a.cols };
}

// Helper function to element-wise multiply
function elementwiseMultiply(a: Tensor, b: Tensor): Tensor {
  const data = a.data.map((row, i) =>
    row.map((val, j) => val * b.data[i][j])
  );
  return { data, rows: a.rows, cols: a.cols };
}

// Helper function to element-wise divide
function elementwiseDivide(a: Tensor, b: Tensor, epsilon: number = 1e-8): Tensor {
  const data = a.data.map((row, i) =>
    row.map((val, j) => val / (b.data[i][j] + epsilon))
  );
  return { data, rows: a.rows, cols: a.cols };
}

// Helper function to element-wise sqrt
function sqrt(t: Tensor): Tensor {
  const data = t.data.map(row => row.map(val => Math.sqrt(val)));
  return { data, rows: t.rows, cols: t.cols };
}

export class SGD implements Optimizer {
  getName(): string {
    return 'SGD (Stochastic Gradient Descent)';
  }

  getFormula(): string {
    return 'w = w - lr × ∇w\nb = b - lr × ∇b';
  }

  update(weights: Tensor, gradients: Tensor, learningRate: number): Tensor {
    const update = multiplyScalar(gradients, learningRate);
    return subtract(weights, update);
  }

  updateBias(bias: number[], gradients: number[], learningRate: number): number[] {
    return bias.map((b, i) => b - learningRate * gradients[i]);
  }

  reset(): void {
    // No state to reset for SGD
  }
}

export class Adam implements Optimizer {
  private m: Map<string, Tensor> = new Map();
  private v: Map<string, Tensor> = new Map();
  private t: number = 0;
  private beta1: number = 0.9;
  private beta2: number = 0.999;
  private epsilon: number = 1e-8;

  private biasM: Map<string, number[]> = new Map();
  private biasV: Map<string, number[]> = new Map();

  getName(): string {
    return 'Adam (Adaptive Moment Estimation)';
  }

  getFormula(): string {
    return 'm = β₁×m + (1-β₁)×∇w\nv = β₂×v + (1-β₂)×∇w²\nm̂ = m / (1-β₁ᵗ)\nv̂ = v / (1-β₂ᵗ)\nw = w - lr × m̂ / (√v̂ + ε)';
  }

  private getKey(weights: Tensor): string {
    return `${weights.rows}x${weights.cols}`;
  }

  private getBiasKey(bias: number[]): string {
    return `${bias.length}`;
  }

  update(weights: Tensor, gradients: Tensor, learningRate: number): Tensor {
    const key = this.getKey(weights);
    this.t += 1;

    // Initialize moment estimates if not present
    if (!this.m.has(key)) {
      this.m.set(key, {
        data: gradients.data.map(row => row.map(() => 0)),
        rows: gradients.rows,
        cols: gradients.cols,
      });
      this.v.set(key, {
        data: gradients.data.map(row => row.map(() => 0)),
        rows: gradients.rows,
        cols: gradients.cols,
      });
    }

    const m = this.m.get(key)!;
    const v = this.v.get(key)!;

    // Update biased first moment estimate
    const mNew = add(
      multiplyScalar(m, this.beta1),
      multiplyScalar(gradients, 1 - this.beta1)
    );
    this.m.set(key, mNew);

    // Update biased second raw moment estimate
    const gradientsSquared = elementwiseMultiply(gradients, gradients);
    const vNew = add(
      multiplyScalar(v, this.beta2),
      multiplyScalar(gradientsSquared, 1 - this.beta2)
    );
    this.v.set(key, vNew);

    // Compute bias-corrected moment estimates
    const mHat = multiplyScalar(mNew, 1 / (1 - Math.pow(this.beta1, this.t)));
    const vHat = multiplyScalar(vNew, 1 / (1 - Math.pow(this.beta2, this.t)));

    // Update weights
    const vHatSqrt = sqrt(vHat);
    const update = elementwiseDivide(multiplyScalar(mHat, learningRate), vHatSqrt, this.epsilon);
    return subtract(weights, update);
  }

  updateBias(bias: number[], gradients: number[], learningRate: number): number[] {
    const key = this.getBiasKey(bias);
    this.t += 1;

    // Initialize moment estimates if not present
    if (!this.biasM.has(key)) {
      this.biasM.set(key, new Array(bias.length).fill(0));
      this.biasV.set(key, new Array(bias.length).fill(0));
    }

    const m = this.biasM.get(key)!;
    const v = this.biasV.get(key)!;

    // Update biased first moment estimate
    const mNew = m.map((mi, i) => this.beta1 * mi + (1 - this.beta1) * gradients[i]);
    this.biasM.set(key, mNew);

    // Update biased second raw moment estimate
    const vNew = v.map((vi, i) => this.beta2 * vi + (1 - this.beta2) * gradients[i] * gradients[i]);
    this.biasV.set(key, vNew);

    // Compute bias-corrected moment estimates
    const mHat = mNew.map(mi => mi / (1 - Math.pow(this.beta1, this.t)));
    const vHat = vNew.map(vi => vi / (1 - Math.pow(this.beta2, this.t)));

    // Update bias
    return bias.map((b, i) => {
      const vHatSqrt = Math.sqrt(vHat[i]);
      return b - learningRate * mHat[i] / (vHatSqrt + this.epsilon);
    });
  }

  reset(): void {
    this.m.clear();
    this.v.clear();
    this.biasM.clear();
    this.biasV.clear();
    this.t = 0;
  }
}

export class RMSprop implements Optimizer {
  private v: Map<string, Tensor> = new Map();
  private beta: number = 0.9;
  private epsilon: number = 1e-8;

  private biasV: Map<string, number[]> = new Map();

  getName(): string {
    return 'RMSprop (Root Mean Square Propagation)';
  }

  getFormula(): string {
    return 'v = β×v + (1-β)×∇w²\nw = w - lr × ∇w / (√v + ε)';
  }

  private getKey(weights: Tensor): string {
    return `${weights.rows}x${weights.cols}`;
  }

  private getBiasKey(bias: number[]): string {
    return `${bias.length}`;
  }

  update(weights: Tensor, gradients: Tensor, learningRate: number): Tensor {
    const key = this.getKey(weights);

    // Initialize v if not present
    if (!this.v.has(key)) {
      this.v.set(key, {
        data: gradients.data.map(row => row.map(() => 0)),
        rows: gradients.rows,
        cols: gradients.cols,
      });
    }

    const v = this.v.get(key)!;

    // Update moving average of squared gradients
    const gradientsSquared = elementwiseMultiply(gradients, gradients);
    const vNew = add(
      multiplyScalar(v, this.beta),
      multiplyScalar(gradientsSquared, 1 - this.beta)
    );
    this.v.set(key, vNew);

    // Update weights
    const vSqrt = sqrt(vNew);
    const update = elementwiseDivide(multiplyScalar(gradients, learningRate), vSqrt, this.epsilon);
    return subtract(weights, update);
  }

  updateBias(bias: number[], gradients: number[], learningRate: number): number[] {
    const key = this.getBiasKey(bias);

    // Initialize v if not present
    if (!this.biasV.has(key)) {
      this.biasV.set(key, new Array(bias.length).fill(0));
    }

    const v = this.biasV.get(key)!;

    // Update moving average of squared gradients
    const vNew = v.map((vi, i) => this.beta * vi + (1 - this.beta) * gradients[i] * gradients[i]);
    this.biasV.set(key, vNew);

    // Update bias
    return bias.map((b, i) => {
      const vSqrt = Math.sqrt(vNew[i]);
      return b - learningRate * gradients[i] / (vSqrt + this.epsilon);
    });
  }

  reset(): void {
    this.v.clear();
    this.biasV.clear();
  }
}

export function getOptimizer(type: 'sgd' | 'adam' | 'rmsprop'): Optimizer {
  switch (type) {
    case 'sgd':
      return new SGD();
    case 'adam':
      return new Adam();
    case 'rmsprop':
      return new RMSprop();
    default:
      return new SGD();
  }
}


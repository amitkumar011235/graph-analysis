/**
 * Loss functions with compute and gradient methods
 */

import { Tensor, LossFunction } from './types';

export class MSE implements LossFunction {
  compute(predictions: Tensor, targets: Tensor): number {
    let sum = 0;
    let count = 0;
    for (let i = 0; i < predictions.rows; i++) {
      for (let j = 0; j < predictions.cols; j++) {
        const pred = predictions.data[i][j];
        const target = targets.data[i][j];
        if (isFinite(pred) && isFinite(target)) {
          const diff = pred - target;
          sum += diff * diff;
          count++;
        }
      }
    }
    return count > 0 ? sum / count : 0;
  }

  gradient(predictions: Tensor, targets: Tensor): Tensor {
    const count = predictions.rows * predictions.cols;
    const data = predictions.data.map((row, i) =>
      row.map((val, j) => {
        const grad = 2 * (val - targets.data[i][j]) / count;
        return isFinite(grad) ? grad : 0;
      })
    );
    return { data, rows: predictions.rows, cols: predictions.cols };
  }
}

export class CrossEntropy implements LossFunction {
  compute(predictions: Tensor, targets: Tensor): number {
    const epsilon = 1e-15;
    let sum = 0;
    let count = 0;
    for (let i = 0; i < predictions.rows; i++) {
      for (let j = 0; j < predictions.cols; j++) {
        const rawPred = predictions.data[i][j];
        if (!isFinite(rawPred)) continue;
        const pred = Math.max(epsilon, Math.min(1 - epsilon, rawPred));
        const target = targets.data[i][j];
        const loss = target * Math.log(pred) + (1 - target) * Math.log(1 - pred);
        if (isFinite(loss)) {
          sum += loss;
          count++;
        }
      }
    }
    return count > 0 ? -sum / count : 0;
  }

  gradient(predictions: Tensor, targets: Tensor): Tensor {
    const epsilon = 1e-15;
    const count = predictions.rows * predictions.cols;
    const data = predictions.data.map((row, i) =>
      row.map((val, j) => {
        const pred = Math.max(epsilon, Math.min(1 - epsilon, val));
        const target = targets.data[i][j];
        const grad = -(target / pred - (1 - target) / (1 - pred)) / count;
        return isFinite(grad) ? Math.max(-10, Math.min(10, grad)) : 0;
      })
    );
    return { data, rows: predictions.rows, cols: predictions.cols };
  }
}

export function getLoss(type: string): LossFunction {
  switch (type) {
    case 'mse':
      return new MSE();
    case 'crossentropy':
      return new CrossEntropy();
    default:
      return new MSE();
  }
}


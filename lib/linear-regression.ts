/**
 * Linear Regression Math Library
 * Provides core functions for normal equation, gradient descent, and loss calculations
 */

export interface DataPoint {
  x: number;
  y: number;
  isTrain?: boolean;
}

export type LossType = 'mse' | 'mae' | 'huber';

/**
 * Normal equation (closed-form solution for linear regression)
 * Returns the optimal m (slope) and b (intercept) that minimize MSE
 */
export function normalEquation(points: DataPoint[]): { m: number; b: number } {
  if (points.length < 2) {
    return { m: 0, b: 0 };
  }

  const n = points.length;
  const sumX = points.reduce((s, p) => s + p.x, 0);
  const sumY = points.reduce((s, p) => s + p.y, 0);
  const sumXY = points.reduce((s, p) => s + p.x * p.y, 0);
  const sumXX = points.reduce((s, p) => s + p.x * p.x, 0);

  const denominator = n * sumXX - sumX * sumX;
  
  if (Math.abs(denominator) < 1e-10) {
    // All points have same x, vertical line case
    return { m: 0, b: sumY / n };
  }

  const m = (n * sumXY - sumX * sumY) / denominator;
  const b = (sumY - m * sumX) / n;

  return { m, b };
}

/**
 * Compute Mean Squared Error
 */
export function computeMSE(points: DataPoint[], m: number, b: number): number {
  if (points.length === 0) return 0;

  let sumSquaredError = 0;
  for (const { x, y } of points) {
    const pred = m * x + b;
    const error = pred - y;
    sumSquaredError += error * error;
  }

  return sumSquaredError / points.length;
}

/**
 * Compute Mean Absolute Error
 */
export function computeMAE(points: DataPoint[], m: number, b: number): number {
  if (points.length === 0) return 0;

  let sumAbsError = 0;
  for (const { x, y } of points) {
    const pred = m * x + b;
    const error = Math.abs(pred - y);
    sumAbsError += error;
  }

  return sumAbsError / points.length;
}

/**
 * Compute Huber Loss
 * Huber loss is less sensitive to outliers than MSE
 */
export function computeHuber(
  points: DataPoint[],
  m: number,
  b: number,
  delta: number = 1.0
): number {
  if (points.length === 0) return 0;

  let sumLoss = 0;
  for (const { x, y } of points) {
    const pred = m * x + b;
    const error = Math.abs(pred - y);
    
    if (error <= delta) {
      sumLoss += 0.5 * error * error;
    } else {
      sumLoss += delta * error - 0.5 * delta * delta;
    }
  }

  return sumLoss / points.length;
}

/**
 * Compute R² (coefficient of determination)
 */
export function computeR2(points: DataPoint[], m: number, b: number): number {
  if (points.length < 2) return 0;

  const meanY = points.reduce((s, p) => s + p.y, 0) / points.length;
  
  let totalSumSquares = 0;
  let residualSumSquares = 0;

  for (const { x, y } of points) {
    const pred = m * x + b;
    const totalError = y - meanY;
    const residualError = y - pred;
    
    totalSumSquares += totalError * totalError;
    residualSumSquares += residualError * residualError;
  }

  if (totalSumSquares < 1e-10) return 1; // Perfect fit

  return 1 - residualSumSquares / totalSumSquares;
}

/**
 * Compute loss based on loss type
 */
export function computeLoss(
  points: DataPoint[],
  m: number,
  b: number,
  lossType: LossType,
  delta?: number
): number {
  switch (lossType) {
    case 'mse':
      return computeMSE(points, m, b);
    case 'mae':
      return computeMAE(points, m, b);
    case 'huber':
      return computeHuber(points, m, b, delta || 1.0);
    default:
      return computeMSE(points, m, b);
  }
}

/**
 * Gradient descent step
 * Returns updated m, b, and gradients
 */
export function gradientDescentStep(
  points: DataPoint[],
  m: number,
  b: number,
  learningRate: number,
  lossType: LossType = 'mse',
  delta: number = 1.0
): { m: number; b: number; gradients: { dm: number; db: number } } {
  if (points.length === 0) {
    return { m, b, gradients: { dm: 0, db: 0 } };
  }

  let dm = 0;
  let db = 0;

  for (const { x, y } of points) {
    const pred = m * x + b;
    const error = pred - y;

    if (lossType === 'mse') {
      // Gradient for MSE: d/dtheta (error²) = 2 * error * d/dtheta(pred)
      dm += 2 * error * x;
      db += 2 * error;
    } else if (lossType === 'mae') {
      // Gradient for MAE: sign(error) * d/dtheta(pred)
      const sign = error >= 0 ? 1 : -1;
      dm += sign * x;
      db += sign;
    } else if (lossType === 'huber') {
      // Gradient for Huber: depends on |error| vs delta
      const absError = Math.abs(error);
      if (absError <= delta) {
        // Quadratic region: same as MSE
        dm += error * x;
        db += error;
      } else {
        // Linear region: sign(error) * delta
        const sign = error >= 0 ? 1 : -1;
        dm += sign * delta * x;
        db += sign * delta;
      }
    }
  }

  const n = points.length;
  const avgDm = dm / n;
  const avgDb = db / n;

  // Apply gradient descent update
  const newM = m - learningRate * avgDm;
  const newB = b - learningRate * avgDb;

  return {
    m: newM,
    b: newB,
    gradients: { dm: avgDm, db: avgDb },
  };
}

/**
 * Compute loss landscape for 3D visualization
 * Returns mesh grids for m, b, and corresponding loss values
 */
export function computeLossLandscape(
  points: DataPoint[],
  mRange: [number, number],
  bRange: [number, number],
  resolution: number = 50,
  lossType: LossType = 'mse',
  delta: number = 1.0
): { m: number[][]; b: number[][]; loss: number[][] } {
  const mMin = mRange[0];
  const mMax = mRange[1];
  const bMin = bRange[0];
  const bMax = bRange[1];

  const mStep = (mMax - mMin) / (resolution - 1);
  const bStep = (bMax - bMin) / (resolution - 1);

  const m: number[][] = [];
  const b: number[][] = [];
  const loss: number[][] = [];

  for (let i = 0; i < resolution; i++) {
    m[i] = [];
    b[i] = [];
    loss[i] = [];
    
    const currentM = mMin + i * mStep;
    
    for (let j = 0; j < resolution; j++) {
      const currentB = bMin + j * bStep;
      
      m[i][j] = currentM;
      b[i][j] = currentB;
      loss[i][j] = computeLoss(points, currentM, currentB, lossType, delta);
    }
  }

  return { m, b, loss };
}


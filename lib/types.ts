/**
 * TypeScript types and interfaces for the neural network library
 */

export type ActivationType = 'relu' | 'sigmoid' | 'tanh' | 'linear' | 'softmax';

export type LossType = 'mse' | 'crossentropy';

export type Mode = 'regression' | 'classification';

export interface DataPoint {
  x: number;
  y: number;
  label?: number; // For classification mode
}

export interface LayerConfig {
  neurons: number;
  activation: ActivationType;
}

export interface NetworkConfig {
  layers: LayerConfig[];
  learningRate: number;
  epochs: number;
}

export interface TrainingState {
  isTraining: boolean;
  currentEpoch: number;
  currentLoss: number;
  lossHistory: number[];
}

export interface Tensor {
  data: number[][];
  rows: number;
  cols: number;
}

export interface ActivationFunction {
  forward(input: Tensor): Tensor;
  backward(outputGrad: Tensor, input: Tensor): Tensor;
}

export interface LossFunction {
  compute(predictions: Tensor, targets: Tensor): number;
  gradient(predictions: Tensor, targets: Tensor): Tensor;
}


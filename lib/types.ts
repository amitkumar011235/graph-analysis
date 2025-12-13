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

// Debug-related types
export type OptimizerType = 'sgd' | 'adam' | 'rmsprop';

export interface NetworkConfig {
  inputSize: number;
  outputSize: number;
  layers: LayerConfig[];
  lossFunction: LossType;
  optimizer: OptimizerType;
  learningRate: number;
  epochs: number;
}

export interface ComputationDetail {
  formula: string;
  description: string;
  inputs: { name: string; value: Tensor | number[] | number }[];
  output: Tensor | number[] | number;
  operation: string;
  layerIndex?: number;
}

export interface StepSnapshot {
  stepType: 'forward' | 'backward' | 'update' | 'loss';
  layerIndex?: number;
  networkState: {
    weights: Tensor[];
    biases: number[][];
    activations?: Tensor[];
  };
  computation: ComputationDetail;
  timestamp: number;
  stepNumber: number;
}

export interface NetworkState {
  weights: Tensor[];
  biases: number[][];
  activations?: Tensor[];
  gradients?: {
    weightGrads: Tensor[];
    biasGrads: number[][];
  };
}


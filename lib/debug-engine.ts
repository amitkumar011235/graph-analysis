/**
 * Step-by-step execution engine for neural network debugging
 */

import { Network, Layer } from './neural-network';
import { Optimizer, getOptimizer } from './optimizers';
import {
  Tensor,
  ComputationDetail,
  StepSnapshot,
  NetworkState,
  NetworkConfig,
  OptimizerType,
} from './types';
import { getLoss } from './losses';

export class DebugEngine {
  private network: Network;
  private optimizer: Optimizer;
  private stepHistory: StepSnapshot[] = [];
  private currentStepIndex: number = -1;
  private currentInput?: Tensor;
  private currentTarget?: Tensor;
  private forwardActivations: Tensor[] = [];
  private currentLoss?: number;
  private lossGradient?: Tensor;

  private currentOptimizerType?: OptimizerType;

  constructor(config: NetworkConfig) {
    // Create network from config
    const layerConfigs = config.layers.map(l => ({
      neurons: l.neurons,
      activation: l.activation,
    }));
    // Ensure last layer matches outputSize
    if (layerConfigs.length > 0) {
      layerConfigs[layerConfigs.length - 1].neurons = config.outputSize;
    }
    this.network = new Network(layerConfigs, config.lossFunction, config.inputSize);
    this.currentOptimizerType = config.optimizer;
    this.optimizer = getOptimizer(config.optimizer);
    this.optimizer.reset();
  }

  // Update config (e.g., when optimizer type changes)
  updateConfig(config: NetworkConfig) {
    // If optimizer type changed, create new optimizer
    if (config.optimizer !== this.currentOptimizerType) {
      this.currentOptimizerType = config.optimizer;
      this.optimizer = getOptimizer(config.optimizer);
      this.optimizer.reset();
    }
  }

  // Initialize with data
  setData(input: Tensor, target: Tensor) {
    this.currentInput = input;
    this.currentTarget = target;
    this.forwardActivations = [];
    this.currentLoss = undefined;
    this.lossGradient = undefined;
    this.stepHistory = [];
    this.currentStepIndex = -1;
  }

  // Get current network state
  getCurrentState(): NetworkState {
    const state = this.network.getState();
    const gradients = this.network.getGradients();
    return {
      ...state,
      activations: this.forwardActivations.length > 0 ? this.forwardActivations : undefined,
      gradients: gradients || undefined,
    };
  }

  // Get step history
  getStepHistory(): StepSnapshot[] {
    return this.stepHistory.slice(0, this.currentStepIndex + 1);
  }

  // Forward pass step: compute pre-activation for a layer
  stepForwardPreActivation(layerIndex: number): ComputationDetail {
    if (!this.currentInput) {
      throw new Error('No input data set');
    }

    const layer = this.network.layers[layerIndex];
    const input = layerIndex === 0 ? this.currentInput : this.forwardActivations[layerIndex - 1];

    // Compute: z = input @ W^T + b
    const weightsT = this.transpose(layer.weights);
    const z = this.matmul(input, weightsT);
    const zWithBias = this.broadcastAdd(z, layer.bias);

    return {
      formula: `z${layerIndex} = X${layerIndex} × W${layerIndex}^T + b${layerIndex}`,
      description: `Forward pass - Layer ${layerIndex}: Pre-activation computation`,
      inputs: [
        { name: `X${layerIndex}`, value: input },
        { name: `W${layerIndex}`, value: layer.weights },
        { name: `b${layerIndex}`, value: layer.bias },
      ],
      output: zWithBias,
      operation: 'linear',
      layerIndex,
    };
  }

  // Forward pass step: apply activation
  stepForwardActivation(layerIndex: number): ComputationDetail {
    if (!this.currentInput) {
      throw new Error('No input data set');
    }

    const layer = this.network.layers[layerIndex];
    const input = layerIndex === 0 ? this.currentInput : this.forwardActivations[layerIndex - 1];

    // Execute forward through layer to get activation
    const activated = this.network.forwardLayer(layerIndex, input);
    
    // Store activation
    if (!this.forwardActivations[layerIndex]) {
      this.forwardActivations[layerIndex] = activated;
    }

    const activationName = this.getActivationName(layer.activationType);

    return {
      formula: `a${layerIndex} = ${activationName}(z${layerIndex})`,
      description: `Forward pass - Layer ${layerIndex}: Apply ${activationName} activation`,
      inputs: [
        { name: `z${layerIndex}`, value: layer.preActivationOutput! },
      ],
      output: activated,
      operation: `activation_${layer.activationType}`,
      layerIndex,
    };
  }

  // Complete forward pass through one layer
  stepForwardLayer(layerIndex: number): StepSnapshot {
    if (!this.currentInput) {
      throw new Error('No input data set');
    }

    const input = layerIndex === 0 ? this.currentInput : this.forwardActivations[layerIndex - 1];
    const output = this.network.forwardLayer(layerIndex, input);
    
    // Store activation
    if (!this.forwardActivations[layerIndex]) {
      this.forwardActivations[layerIndex] = output;
    } else {
      this.forwardActivations[layerIndex] = output;
    }

    const layer = this.network.layers[layerIndex];
    const activationName = this.getActivationName(layer.activationType);

    const computation: ComputationDetail = {
      formula: `z${layerIndex} = X${layerIndex} × W${layerIndex}^T + b${layerIndex}\na${layerIndex} = ${activationName}(z${layerIndex})`,
      description: `Forward pass through Layer ${layerIndex}`,
      inputs: [
        { name: `Input`, value: input },
        { name: `Weights`, value: layer.weights },
        { name: `Bias`, value: layer.bias },
      ],
      output: output,
      operation: 'forward',
      layerIndex,
    };

    const snapshot: StepSnapshot = {
      stepType: 'forward',
      layerIndex,
      networkState: this.getCurrentState(),
      computation,
      timestamp: Date.now(),
      stepNumber: this.currentStepIndex + 1,
    };

    this.currentStepIndex++;
    this.stepHistory = this.stepHistory.slice(0, this.currentStepIndex);
    this.stepHistory.push(snapshot);

    return snapshot;
  }

  // Compute loss
  stepComputeLoss(): StepSnapshot {
    if (!this.currentInput || !this.currentTarget) {
      throw new Error('Input and target data must be set');
    }

    // Complete forward pass if not done
    if (this.forwardActivations.length < this.network.layers.length) {
      let current = this.currentInput;
      for (let i = 0; i < this.network.layers.length; i++) {
        current = this.network.forwardLayer(i, current);
        this.forwardActivations[i] = current;
      }
    }

    const predictions = this.forwardActivations[this.forwardActivations.length - 1];
    const loss = this.network.lossFunction.compute(predictions, this.currentTarget);
    this.currentLoss = loss;

    const lossName = this.getLossName();
    const computation: ComputationDetail = {
      formula: `L = ${lossName}(y_pred, y_true)`,
      description: 'Compute loss',
      inputs: [
        { name: 'Predictions', value: predictions },
        { name: 'Targets', value: this.currentTarget },
      ],
      output: loss,
      operation: 'loss',
    };

    const snapshot: StepSnapshot = {
      stepType: 'loss',
      networkState: this.getCurrentState(),
      computation,
      timestamp: Date.now(),
      stepNumber: this.currentStepIndex + 1,
    };

    this.currentStepIndex++;
    this.stepHistory = this.stepHistory.slice(0, this.currentStepIndex);
    this.stepHistory.push(snapshot);

    return snapshot;
  }

  // Backward pass step: compute loss gradient
  stepBackwardLossGradient(): StepSnapshot {
    if (!this.currentTarget || this.currentLoss === undefined) {
      throw new Error('Must compute loss first');
    }

    const predictions = this.forwardActivations[this.forwardActivations.length - 1];
    const grad = this.network.lossFunction.gradient(predictions, this.currentTarget);
    this.lossGradient = grad;

    const lossName = this.getLossName();
    const computation: ComputationDetail = {
      formula: `∂L/∂a = ∇${lossName}(y_pred, y_true)`,
      description: 'Compute loss gradient',
      inputs: [
        { name: 'Predictions', value: predictions },
        { name: 'Targets', value: this.currentTarget },
      ],
      output: grad,
      operation: 'loss_gradient',
    };

    const snapshot: StepSnapshot = {
      stepType: 'backward',
      networkState: this.getCurrentState(),
      computation,
      timestamp: Date.now(),
      stepNumber: this.currentStepIndex + 1,
    };

    this.currentStepIndex++;
    this.stepHistory = this.stepHistory.slice(0, this.currentStepIndex);
    this.stepHistory.push(snapshot);

    return snapshot;
  }

  // Backward pass step: backprop through a layer
  stepBackwardLayer(layerIndex: number): StepSnapshot {
    if (this.lossGradient === undefined) {
      throw new Error('Must compute loss gradient first');
    }

    const layer = this.network.layers[layerIndex];
    
    // Get the gradient flowing into this layer
    let inputGrad: Tensor;
    if (layerIndex === this.network.layers.length - 1) {
      inputGrad = this.lossGradient;
    } else {
      // This would be computed by the previous backward step
      // For now, we need to do the backward pass properly
      throw new Error('Backward pass must be done in reverse order');
    }

    // Execute backward pass
    const outputGrad = layer.backward(inputGrad);

    const activationName = this.getActivationName(layer.activationType);

    const computation: ComputationDetail = {
      formula: `∂L/∂W${layerIndex} = (∂L/∂a${layerIndex})^T × X${layerIndex}\n∂L/∂b${layerIndex} = sum(∂L/∂a${layerIndex})\n∂L/∂X${layerIndex} = ∂L/∂a${layerIndex} × W${layerIndex}`,
      description: `Backward pass through Layer ${layerIndex}`,
      inputs: [
        { name: `∂L/∂a${layerIndex}`, value: inputGrad },
        { name: `X${layerIndex}`, value: layer.lastInput! },
        { name: `W${layerIndex}`, value: layer.weights },
      ],
      output: outputGrad,
      operation: 'backward',
      layerIndex,
    };

    const snapshot: StepSnapshot = {
      stepType: 'backward',
      layerIndex,
      networkState: this.getCurrentState(),
      computation,
      timestamp: Date.now(),
      stepNumber: this.currentStepIndex + 1,
    };

    this.currentStepIndex++;
    this.stepHistory = this.stepHistory.slice(0, this.currentStepIndex);
    this.stepHistory.push(snapshot);

    return snapshot;
  }

  // Complete backward pass (helper to do it properly in reverse)
  stepBackwardComplete(): StepSnapshot[] {
    if (this.lossGradient === undefined) {
      // Compute loss gradient first
      this.stepBackwardLossGradient();
    }

    const snapshots: StepSnapshot[] = [];
    let currentGrad = this.lossGradient!;

    // Backward through layers in reverse order
    for (let i = this.network.layers.length - 1; i >= 0; i--) {
      const layer = this.network.layers[i];
      currentGrad = layer.backward(currentGrad);
      
      const activationName = this.getActivationName(layer.activationType);
      const computation: ComputationDetail = {
        formula: `∂L/∂W${i} = (∂L/∂a${i})^T × X${i}\n∂L/∂b${i} = sum(∂L/∂a${i})\n∂L/∂X${i} = ∂L/∂a${i} × W${i}`,
        description: `Backward pass through Layer ${i}`,
        inputs: [
          { name: `Gradient`, value: currentGrad },
          { name: `Input`, value: layer.lastInput! },
          { name: `Weights`, value: layer.weights },
        ],
        output: layer.weightGrad!,
        operation: 'backward',
        layerIndex: i,
      };

      const snapshot: StepSnapshot = {
        stepType: 'backward',
        layerIndex: i,
        networkState: this.getCurrentState(),
        computation,
        timestamp: Date.now(),
        stepNumber: this.currentStepIndex + 1,
      };

      this.currentStepIndex++;
      this.stepHistory = this.stepHistory.slice(0, this.currentStepIndex);
      this.stepHistory.push(snapshot);
      snapshots.push(snapshot);
    }

    return snapshots;
  }

  // Weight update step
  stepUpdateWeights(learningRate: number): StepSnapshot {
    const computations: ComputationDetail[] = [];

    for (let i = 0; i < this.network.layers.length; i++) {
      const layer = this.network.layers[i];
      if (!layer.weightGrad || !layer.biasGrad) {
        throw new Error('Gradients must be computed before weight update');
      }

      // Store old weights
      const oldWeights = JSON.parse(JSON.stringify(layer.weights));
      const oldBias = [...layer.bias];

      // Update using optimizer
      layer.weights = this.optimizer.update(layer.weights, layer.weightGrad, learningRate);
      layer.bias = this.optimizer.updateBias(layer.bias, layer.biasGrad, learningRate);

      const optimizerName = this.optimizer.getName();
      computations.push({
        formula: this.optimizer.getFormula(),
        description: `Update weights for Layer ${i} using ${optimizerName}`,
        inputs: [
          { name: 'Old Weights', value: oldWeights },
          { name: 'Weight Gradients', value: layer.weightGrad },
          { name: 'Old Bias', value: oldBias },
          { name: 'Bias Gradients', value: layer.biasGrad },
          { name: 'Learning Rate', value: learningRate },
        ],
        output: layer.weights,
        operation: 'update',
        layerIndex: i,
      });
    }

    // Get first layer's updated weights as representation
    const representativeOutput = this.network.layers[0].weights;
    
    const computation: ComputationDetail = {
      formula: this.optimizer.getFormula(),
      description: `Update all weights using ${this.optimizer.getName()}`,
      inputs: computations.flatMap(c => c.inputs),
      output: representativeOutput, // Use representative tensor for display
      operation: 'update',
    };

    const snapshot: StepSnapshot = {
      stepType: 'update',
      networkState: this.getCurrentState(),
      computation,
      timestamp: Date.now(),
      stepNumber: this.currentStepIndex + 1,
    };

    this.currentStepIndex++;
    this.stepHistory = this.stepHistory.slice(0, this.currentStepIndex);
    this.stepHistory.push(snapshot);

    return snapshot;
  }

  // Undo last step
  undo(): boolean {
    if (this.currentStepIndex <= 0) {
      return false;
    }
    this.currentStepIndex--;
    this.restoreState(this.stepHistory[this.currentStepIndex]);
    return true;
  }

  // Redo next step
  redo(): boolean {
    if (this.currentStepIndex >= this.stepHistory.length - 1) {
      return false;
    }
    this.currentStepIndex++;
    this.restoreState(this.stepHistory[this.currentStepIndex]);
    return true;
  }

  // Restore network state from snapshot
  private restoreState(snapshot: StepSnapshot) {
    const { weights, biases } = snapshot.networkState;
    for (let i = 0; i < this.network.layers.length; i++) {
      this.network.layers[i].weights = weights[i];
      this.network.layers[i].bias = biases[i];
    }
    if (snapshot.networkState.activations) {
      this.forwardActivations = snapshot.networkState.activations;
    }
  }

  // Helper methods
  private matmul(a: Tensor, b: Tensor): Tensor {
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

  private transpose(t: Tensor): Tensor {
    const data: number[][] = [];
    for (let j = 0; j < t.cols; j++) {
      data[j] = [];
      for (let i = 0; i < t.rows; i++) {
        data[j][i] = t.data[i][j];
      }
    }
    return { data, rows: t.cols, cols: t.rows };
  }

  private broadcastAdd(t: Tensor, vec: number[]): Tensor {
    const data = t.data.map((row, i) =>
      row.map((val, j) => val + vec[j])
    );
    return { data, rows: t.rows, cols: t.cols };
  }

  private getActivationName(type: string): string {
    const names: Record<string, string> = {
      relu: 'ReLU',
      sigmoid: 'σ',
      tanh: 'tanh',
      linear: 'linear',
      softmax: 'softmax',
    };
    return names[type] || type;
  }

  private getLossName(): string {
    // This is a simplified version - in practice we'd get it from network
    return 'Loss';
  }

  // Get network instance (for external access if needed)
  getNetwork(): Network {
    return this.network;
  }

  // Get optimizer instance (for training)
  getOptimizer(): Optimizer {
    return this.optimizer;
  }

  // Train for one epoch (for epoch-by-epoch training)
  trainOneEpoch(input: Tensor, target: Tensor, learningRate: number): number {
    // Forward pass
    const predictions = this.network.forward(input);

    // Compute loss
    const loss = this.network.lossFunction.compute(predictions, target);
    this.currentLoss = loss;

    // Backward pass
    const grad = this.network.lossFunction.gradient(predictions, target);
    this.network.backward(grad);

    // Update weights using optimizer
    for (const layer of this.network.layers) {
      if (layer.weightGrad && layer.biasGrad) {
        layer.weights = this.optimizer.update(layer.weights, layer.weightGrad, learningRate);
        layer.bias = this.optimizer.updateBias(layer.bias, layer.biasGrad, learningRate);
      }
    }

    return loss;
  }
}


'use client';

import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { DebugEngine } from '@/lib/debug-engine';
import { NetworkConfig, DataPoint, LayerConfig, ActivationType, LossType, OptimizerType, Tensor } from '@/lib/types';
import NetworkGraph from './NetworkGraph';
import MatrixView from './MatrixView';
import DebugControls from './DebugControls';
import DataCanvas from './DataCanvas';

export default function NeuralNetworkDebugger() {
  const [networkConfig, setNetworkConfig] = useState<NetworkConfig>({
    inputSize: 1,
    outputSize: 1,
    layers: [
      { neurons: 8, activation: 'relu' },
      { neurons: 4, activation: 'relu' },
      { neurons: 1, activation: 'linear' },
    ],
    lossFunction: 'mse',
    optimizer: 'sgd',
    learningRate: 0.01,
    epochs: 10,
  });

  const [dataPoints, setDataPoints] = useState<DataPoint[]>([]);
  const [debugEngine, setDebugEngine] = useState<DebugEngine | null>(null);
  const [currentStep, setCurrentStep] = useState(0);
  const [selectedLayer, setSelectedLayer] = useState<number | undefined>(undefined);
  const [selectedNode, setSelectedNode] = useState<{ layerIndex: number; nodeIndex?: number } | null>(null);
  const [isAutoStepping, setIsAutoStepping] = useState(false);
  const autoStepIntervalRef = useRef<NodeJS.Timeout | null>(null);
  
  // Training state
  const [isTraining, setIsTraining] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [predictionCurve, setPredictionCurve] = useState<Array<{ x: number; y: number }>>([]);
  const [currentLoss, setCurrentLoss] = useState<number | undefined>(undefined);
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  
  // Training control refs (to communicate with the training loop)
  const shouldStopRef = useRef(false);
  const isPausedRef = useRef(false);
  const pausedAtEpochRef = useRef(0);
  const trainingTensorsRef = useRef<{ input: Tensor; target: Tensor } | null>(null);
  const lossHistoryRef = useRef<number[]>([]);

  // Initialize debug engine when config changes
  useEffect(() => {
    const engine = new DebugEngine(networkConfig);
    setDebugEngine(engine);
    setCurrentStep(0);
    setSelectedLayer(undefined);
    setSelectedNode(null);
    setPredictionCurve([]);
    setCurrentLoss(undefined);
    setLossHistory([]);
    setCurrentEpoch(0);
    setIsPaused(false);
    shouldStopRef.current = true; // Stop any running training
    isPausedRef.current = false;
    pausedAtEpochRef.current = 0;
    trainingTensorsRef.current = null;
    lossHistoryRef.current = [];
  }, [networkConfig]);

  // Convert data points to tensors
  const dataToTensors = useCallback((points: DataPoint[]): { input: Tensor; target: Tensor } | null => {
    if (points.length === 0) return null;

    const inputData: number[][] = points.map(p => [p.x]);
    const targetData: number[][] = points.map(p => [p.y]);

    return {
      input: { data: inputData, rows: points.length, cols: 1 },
      target: { data: targetData, rows: points.length, cols: 1 },
    };
  }, []);

  // Initialize data in debug engine when points change
  useEffect(() => {
    if (!debugEngine || dataPoints.length === 0) return;

    const tensors = dataToTensors(dataPoints);
    if (tensors) {
      debugEngine.setData(tensors.input, tensors.target);
      setCurrentStep(0);
    }
  }, [debugEngine, dataPoints, dataToTensors]);

  // Get current phase and computation
  const { currentPhase, currentComputation, canGoForward, canGoBack } = useMemo(() => {
    if (!debugEngine || dataPoints.length === 0) {
      return {
        currentPhase: 'idle' as const,
        currentComputation: undefined,
        canGoForward: false,
        canGoBack: false,
      };
    }

    const history = debugEngine.getStepHistory();
    const currentSnapshot = history[currentStep];

    // Determine phase based on step history
    let phase: 'idle' | 'forward' | 'backward' | 'update' | 'completed' = 'idle';
    if (currentSnapshot) {
      phase = currentSnapshot.stepType === 'forward' ? 'forward' :
              currentSnapshot.stepType === 'backward' ? 'backward' :
              currentSnapshot.stepType === 'update' ? 'update' : 'idle';
    } else if (history.length === 0) {
      phase = 'forward'; // Ready to start forward pass
    }

    // Check if we can go forward (need data and haven't completed)
    const totalForwardSteps = networkConfig.layers.length;
    const totalBackwardSteps = networkConfig.layers.length + 1; // +1 for loss gradient
    const totalSteps = totalForwardSteps + totalBackwardSteps + 1; // +1 for weight update
    const canGoForward = currentStep < totalSteps - 1 && dataPoints.length > 0;
    const canGoBack = currentStep > 0;

    return {
      currentPhase: phase,
      currentComputation: currentSnapshot?.computation,
      canGoForward,
      canGoBack,
    };
  }, [debugEngine, currentStep, dataPoints.length, networkConfig.layers.length]);

  // Next step handler
  const handleNextStep = useCallback(() => {
    if (!debugEngine || dataPoints.length === 0) return;

    const tensors = dataToTensors(dataPoints);
    if (!tensors) return;

    const history = debugEngine.getStepHistory();
    const totalForwardSteps = networkConfig.layers.length;
    const totalBackwardSteps = networkConfig.layers.length + 1;

    try {
      // Forward pass steps
      if (currentStep < totalForwardSteps) {
        const layerIndex = currentStep;
        debugEngine.stepForwardLayer(layerIndex);
        setCurrentStep(prev => prev + 1);
        setSelectedLayer(layerIndex);
      }
      // Loss computation
      else if (currentStep === totalForwardSteps) {
        debugEngine.stepComputeLoss();
        setCurrentStep(prev => prev + 1);
      }
      // Loss gradient
      else if (currentStep === totalForwardSteps + 1) {
        debugEngine.stepBackwardLossGradient();
        setCurrentStep(prev => prev + 1);
      }
      // Backward pass steps - simplified approach
      else if (currentStep < totalForwardSteps + 1 + totalBackwardSteps) {
        // For now, execute complete backward pass
        // In production, this would be done step-by-step per layer
        const network = debugEngine.getNetwork();
        const predictions = network.forward(tensors.input);
        const lossGrad = network.lossFunction.gradient(predictions, tensors.target);
        
        // Execute backward through all layers (simplified)
        let grad = lossGrad;
        for (let i = networkConfig.layers.length - 1; i >= 0; i--) {
          grad = network.layers[i].backward(grad);
          if (i === networkConfig.layers.length - 1 - (currentStep - totalForwardSteps - 2)) {
            setSelectedLayer(i);
            break;
          }
        }
        
        // Create snapshot manually (simplified)
        setCurrentStep(prev => prev + 1);
      }
      // Weight update
      else if (currentStep === totalForwardSteps + 1 + totalBackwardSteps) {
        debugEngine.stepUpdateWeights(networkConfig.learningRate);
        setCurrentStep(prev => prev + 1);
      }
    } catch (error) {
      console.error('Error executing step:', error);
    }
  }, [debugEngine, dataPoints, currentStep, networkConfig, dataToTensors]);

  // Previous step handler
  const handlePreviousStep = useCallback(() => {
    if (!debugEngine) return;
    
    if (debugEngine.undo()) {
      setCurrentStep(prev => Math.max(0, prev - 1));
    }
  }, [debugEngine]);

  // Reset handler
  const handleReset = useCallback(() => {
    if (debugEngine && dataPoints.length > 0) {
      const tensors = dataToTensors(dataPoints);
      if (tensors) {
        debugEngine.setData(tensors.input, tensors.target);
      }
    }
    setCurrentStep(0);
    setSelectedLayer(undefined);
    setSelectedNode(null);
    setIsAutoStepping(false);
  }, [debugEngine, dataPoints, dataToTensors]);

  // Auto step handler
  const handleAutoStep = useCallback((enabled: boolean) => {
    setIsAutoStepping(enabled);
  }, []);

  // Auto step effect
  useEffect(() => {
    if (isAutoStepping && canGoForward) {
      autoStepIntervalRef.current = setInterval(() => {
        handleNextStep();
      }, 500); // Step every 500ms
    } else {
      if (autoStepIntervalRef.current) {
        clearInterval(autoStepIntervalRef.current);
        autoStepIntervalRef.current = null;
      }
    }

    return () => {
      if (autoStepIntervalRef.current) {
        clearInterval(autoStepIntervalRef.current);
      }
    };
  }, [isAutoStepping, canGoForward, handleNextStep]);

  // Handle node click
  const handleNodeClick = useCallback((layerIndex: number, nodeIndex?: number) => {
    setSelectedNode({ layerIndex, nodeIndex });
    setSelectedLayer(layerIndex);
  }, []);

  // Get selected layer details for matrix view
  const selectedLayerDetails = useMemo(() => {
    if (selectedLayer === undefined || !debugEngine) return null;

    const network = debugEngine.getNetwork();
    const state = debugEngine.getCurrentState();

    return {
      weights: state.weights[selectedLayer],
      bias: state.biases[selectedLayer],
      activations: state.activations?.[selectedLayer],
      gradients: state.gradients ? {
        weightGrad: state.gradients.weightGrads[selectedLayer],
        biasGrad: state.gradients.biasGrads[selectedLayer],
      } : undefined,
    };
  }, [selectedLayer, debugEngine]);

  // Configuration handlers
  const handleAddLayer = () => {
    if (networkConfig.layers.length < 5) {
      setNetworkConfig(prev => {
        const newLayers = [...prev.layers, { neurons: prev.outputSize, activation: 'relu' as ActivationType }];
        return {
          ...prev,
          layers: newLayers,
        };
      });
    }
  };

  const handleRemoveLayer = (index: number) => {
    if (networkConfig.layers.length > 1) {
      setNetworkConfig(prev => {
        const newLayers = prev.layers.filter((_, i) => i !== index);
        // If the last layer was removed, update outputSize to match the new last layer
        const updatedOutputSize = newLayers.length > 0 
          ? newLayers[newLayers.length - 1].neurons 
          : prev.outputSize;
        return {
          ...prev,
          layers: newLayers,
          outputSize: updatedOutputSize,
        };
      });
    }
  };

  const handleLayerChange = (index: number, field: 'neurons' | 'activation', value: number | ActivationType) => {
    setNetworkConfig(prev => {
      const newLayers = prev.layers.map((layer, i) =>
        i === index ? { ...layer, [field]: value } : layer
      );
      
      // If the last layer's neurons changed, update outputSize
      const isLastLayer = index === prev.layers.length - 1;
      const updatedOutputSize = isLastLayer && field === 'neurons' 
        ? (value as number) 
        : prev.outputSize;
      
      return {
        ...prev,
        layers: newLayers,
        outputSize: updatedOutputSize,
      };
    });
  };
  
  const handleOutputSizeChange = (outputSize: number) => {
    setNetworkConfig(prev => {
      // Update last layer's neurons to match output size
      const newLayers = [...prev.layers];
      if (newLayers.length > 0) {
        newLayers[newLayers.length - 1] = {
          ...newLayers[newLayers.length - 1],
          neurons: outputSize,
        };
      }
      
      return {
        ...prev,
        outputSize,
        layers: newLayers,
      };
    });
  };

  // Data generators
  const generateLinearData = () => {
    const points: DataPoint[] = [];
    for (let i = 0; i < 30; i++) {
      const x = -5 + (i / 29) * 10;
      const y = 2 * x + 3;
      points.push({ x, y });
    }
    setDataPoints(points);
  };

  const generateNoisyLinearData = () => {
    const points: DataPoint[] = [];
    for (let i = 0; i < 40; i++) {
      const x = -5 + (i / 39) * 10;
      const y = 2 * x + 3 + (Math.random() - 0.5) * 3;
      points.push({ x, y });
    }
    setDataPoints(points);
  };

  const generateQuadraticData = () => {
    const points: DataPoint[] = [];
    for (let i = 0; i < 40; i++) {
      const x = -5 + (i / 39) * 10;
      const y = 0.3 * x * x - 2 * x + 1 + (Math.random() - 0.5) * 1;
      points.push({ x, y });
    }
    setDataPoints(points);
  };

  const generateSineData = () => {
    const points: DataPoint[] = [];
    for (let i = 0; i < 50; i++) {
      const x = -5 + (i / 49) * 10;
      const y = 3 * Math.sin(x) + 2 + (Math.random() - 0.5) * 0.5;
      points.push({ x, y });
    }
    setDataPoints(points);
  };

  const generateRandomScatteredData = () => {
    const points: DataPoint[] = [];
    for (let i = 0; i < 50; i++) {
      const x = -5 + Math.random() * 10;
      const y = -5 + Math.random() * 10;
      points.push({ x, y });
    }
    setDataPoints(points);
  };

  const generateXORData = () => {
    // Generate XOR-like data for classification testing
    const points: DataPoint[] = [];
    // Class 0
    for (let i = 0; i < 20; i++) {
      points.push({ x: -3 + Math.random() * 2, y: -3 + Math.random() * 2, label: 0 });
    }
    for (let i = 0; i < 20; i++) {
      points.push({ x: 1 + Math.random() * 2, y: 1 + Math.random() * 2, label: 0 });
    }
    // Class 1
    for (let i = 0; i < 20; i++) {
      points.push({ x: -3 + Math.random() * 2, y: 1 + Math.random() * 2, label: 1 });
    }
    for (let i = 0; i < 20; i++) {
      points.push({ x: 1 + Math.random() * 2, y: -3 + Math.random() * 2, label: 1 });
    }
    setDataPoints(points);
  };

  const clearAllData = () => {
    setDataPoints([]);
  };

  // Compute bounds for data canvas
  const bounds = useMemo(() => {
    if (dataPoints.length === 0) {
      return { xMin: -5, xMax: 5, yMin: -5, yMax: 5 };
    }

    let xMin = Math.min(...dataPoints.map(p => p.x));
    let xMax = Math.max(...dataPoints.map(p => p.x));
    let yMin = Math.min(...dataPoints.map(p => p.y));
    let yMax = Math.max(...dataPoints.map(p => p.y));

    const xRange = xMax - xMin || 10;
    const yRange = yMax - yMin || 10;

    return {
      xMin: xMin - xRange * 0.2,
      xMax: xMax + xRange * 0.2,
      yMin: yMin - yRange * 0.2,
      yMax: yMax + yRange * 0.2,
    };
  }, [dataPoints]);

  // Generate prediction curve from network (returns curve directly for use in training loop)
  const computePredictionCurve = useCallback((engine: DebugEngine, currentBounds: typeof bounds): Array<{ x: number; y: number }> => {
    const network = engine.getNetwork();
    const resolution = 200; // Number of points for smooth curve
    const curve: Array<{ x: number; y: number }> = [];

    // Get x range from bounds
    const xRange = currentBounds.xMax - currentBounds.xMin;

    for (let i = 0; i <= resolution; i++) {
      const x = currentBounds.xMin + (i / resolution) * xRange;
      
      try {
        // Predict y for this x value
        const input: Tensor = { data: [[x]], rows: 1, cols: 1 };
        const output = network.forward(input);
        const y = output.data[0][0];
        
        // Only add valid predictions
        if (isFinite(y)) {
          curve.push({ x, y });
        }
      } catch (error) {
        // Skip invalid predictions
        console.warn('Error predicting for x:', x, error);
      }
    }

    return curve;
  }, []);

  // Update prediction curve when network or data changes (not during training)
  useEffect(() => {
    if (!isTraining && debugEngine && dataPoints.length > 0) {
      const curve = computePredictionCurve(debugEngine, bounds);
      setPredictionCurve(curve);
    } else if (dataPoints.length === 0) {
      setPredictionCurve([]);
    }
  }, [debugEngine, dataPoints.length, bounds, isTraining, computePredictionCurve]);

  // Training function - runs multiple epochs with proper DNN calculations
  const trainNetwork = useCallback(async (resumeFromEpoch: number = 0) => {
    if (!debugEngine || dataPoints.length === 0) return;

    // Reset control flags
    shouldStopRef.current = false;
    isPausedRef.current = false;

    setIsTraining(true);
    setIsPaused(false);

    // Get or reuse tensors
    let tensors = trainingTensorsRef.current;
    if (!tensors || resumeFromEpoch === 0) {
      tensors = dataToTensors(dataPoints);
      if (!tensors) {
        setIsTraining(false);
        return;
      }
      trainingTensorsRef.current = tensors;
    }

    // Initialize or continue loss history
    let losses: number[] = resumeFromEpoch > 0 ? [...lossHistoryRef.current] : [];
    if (resumeFromEpoch === 0) {
      setCurrentEpoch(0);
      setPredictionCurve([]);
      setLossHistory([]);
      lossHistoryRef.current = [];
    }

    console.log(`${resumeFromEpoch > 0 ? 'Resuming' : 'Starting'} training from epoch ${resumeFromEpoch + 1}:`, {
      epochs: networkConfig.epochs,
      learningRate: networkConfig.learningRate,
      dataPoints: dataPoints.length,
    });

    try {
      for (let epoch = resumeFromEpoch; epoch < networkConfig.epochs; epoch++) {
        // Check for stop signal
        if (shouldStopRef.current) {
          console.log('Training stopped by user');
          break;
        }

        // Check for pause signal
        if (isPausedRef.current) {
          pausedAtEpochRef.current = epoch;
          console.log(`Training paused at epoch ${epoch + 1}`);
          setIsTraining(false);
          setIsPaused(true);
          return; // Exit but keep state for resume
        }

        // Train one epoch (handles forward, backward, and weight update)
        const loss = debugEngine.trainOneEpoch(tensors.input, tensors.target, networkConfig.learningRate);
        losses.push(loss);
        lossHistoryRef.current = [...losses];
        
        // Update state for UI
        setCurrentEpoch(epoch + 1);
        setCurrentLoss(loss);
        setLossHistory([...losses]);

        // Generate and set prediction curve directly after each epoch
        const curve = computePredictionCurve(debugEngine, bounds);
        setPredictionCurve(curve);

        console.log(`Epoch ${epoch + 1}/${networkConfig.epochs}: Loss = ${loss.toFixed(6)}`);

        // Delay for UI update - use longer delay for visual effect
        await new Promise(resolve => setTimeout(resolve, 100));
      }
    } catch (error) {
      console.error('Training error:', error);
    } finally {
      if (!isPausedRef.current) {
        setIsTraining(false);
        setIsPaused(false);
        // Final prediction curve update
        const finalCurve = computePredictionCurve(debugEngine, bounds);
        setPredictionCurve(finalCurve);
      }
    }
  }, [debugEngine, dataPoints, networkConfig, bounds, dataToTensors, computePredictionCurve]);

  // Pause training
  const pauseTraining = useCallback(() => {
    isPausedRef.current = true;
  }, []);

  // Resume training from paused state
  const resumeTraining = useCallback(() => {
    if (isPaused && pausedAtEpochRef.current < networkConfig.epochs) {
      trainNetwork(pausedAtEpochRef.current);
    }
  }, [isPaused, networkConfig.epochs, trainNetwork]);

  // Stop training completely
  const stopTraining = useCallback(() => {
    shouldStopRef.current = true;
    isPausedRef.current = false;
    setIsTraining(false);
    setIsPaused(false);
    pausedAtEpochRef.current = 0;
    trainingTensorsRef.current = null;
    lossHistoryRef.current = [];
  }, []);

  const totalSteps = networkConfig.layers.length * 2 + 2; // forward + backward + loss + update

  return (
    <div className="h-full flex flex-col bg-gray-50">
      <div className="flex-1 flex gap-3 p-3 min-h-0 overflow-hidden">
        {/* Configuration Panel */}
        <div className="w-72 flex-shrink-0 overflow-y-auto bg-white rounded-lg shadow-sm p-4">
          <h2 className="text-lg font-bold text-gray-900 mb-4">Network Configuration</h2>

          {/* Input Size */}
          <div className="mb-4">
            <label className="block text-sm font-semibold text-gray-900 mb-2">Input Size</label>
            <input
              type="number"
              min="1"
              max="10"
              value={networkConfig.inputSize}
              onChange={(e) => setNetworkConfig(prev => ({ ...prev, inputSize: parseInt(e.target.value) || 1 }))}
              className="w-full px-3 py-2 border border-gray-300 rounded text-gray-900 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {/* Output Size */}
          <div className="mb-4">
            <label className="block text-sm font-semibold text-gray-900 mb-2">Output Size</label>
            <input
              type="number"
              min="1"
              max="10"
              value={networkConfig.outputSize}
              onChange={(e) => handleOutputSizeChange(parseInt(e.target.value) || 1)}
              className="w-full px-3 py-2 border border-gray-300 rounded text-gray-900 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <p className="text-xs text-gray-600 mt-1">Will update the last layer's neurons</p>
          </div>

          {/* Layers */}
          <div className="mb-4">
            <div className="flex items-center justify-between mb-2">
              <label className="block text-sm font-semibold text-gray-900">Layers</label>
              <button
                onClick={handleAddLayer}
                disabled={networkConfig.layers.length >= 5}
                className="px-2 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
              >
                + Add
              </button>
            </div>
            <div className="space-y-3">
              {networkConfig.layers.map((layer, index) => (
                <div key={index} className="border border-gray-200 rounded p-3 bg-gray-50">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-semibold text-gray-900">Layer {index + 1}</span>
                    <button
                      onClick={() => handleRemoveLayer(index)}
                      disabled={networkConfig.layers.length <= 1}
                      className="px-2 py-1 text-xs bg-red-600 text-white rounded hover:bg-red-700 disabled:opacity-50"
                    >
                      Remove
                    </button>
                  </div>
                  <div className="space-y-2">
                    <div>
                      <label className="block text-xs font-medium text-gray-800 mb-1">Neurons</label>
                      <input
                        type="number"
                        min="1"
                        max="32"
                        value={layer.neurons}
                        onChange={(e) => handleLayerChange(index, 'neurons', parseInt(e.target.value) || 1)}
                        className="w-full px-2 py-1 text-sm border border-gray-300 rounded text-gray-900 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                      />
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-gray-800 mb-1">Activation</label>
                      <select
                        value={layer.activation}
                        onChange={(e) => handleLayerChange(index, 'activation', e.target.value as ActivationType)}
                        className="w-full px-2 py-1 text-sm border border-gray-300 rounded text-gray-900 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                      >
                        <option value="relu">ReLU</option>
                        <option value="sigmoid">Sigmoid</option>
                        <option value="tanh">Tanh</option>
                        <option value="linear">Linear</option>
                        <option value="softmax">Softmax</option>
                      </select>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Loss Function */}
          <div className="mb-4">
            <label className="block text-sm font-semibold text-gray-900 mb-2">Loss Function</label>
            <select
              value={networkConfig.lossFunction}
              onChange={(e) => setNetworkConfig(prev => ({ ...prev, lossFunction: e.target.value as LossType }))}
              className="w-full px-3 py-2 border border-gray-300 rounded text-gray-900 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="mse">MSE (Mean Squared Error)</option>
              <option value="crossentropy">Cross Entropy</option>
            </select>
          </div>

          {/* Optimizer */}
          <div className="mb-4">
            <label className="block text-sm font-semibold text-gray-900 mb-2">Optimizer</label>
            <select
              value={networkConfig.optimizer}
              onChange={(e) => setNetworkConfig(prev => ({ ...prev, optimizer: e.target.value as OptimizerType }))}
              className="w-full px-3 py-2 border border-gray-300 rounded text-gray-900 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="sgd">SGD</option>
              <option value="adam">Adam</option>
              <option value="rmsprop">RMSprop</option>
            </select>
          </div>

          {/* Learning Rate */}
          <div className="mb-4">
            <label className="block text-sm font-semibold text-gray-900 mb-2">
              Learning Rate: <span className="text-blue-600 font-mono">{networkConfig.learningRate.toFixed(4)}</span>
            </label>
            <input
              type="range"
              min="0.0001"
              max="0.1"
              step="0.0001"
              value={networkConfig.learningRate}
              onChange={(e) => setNetworkConfig(prev => ({ ...prev, learningRate: parseFloat(e.target.value) }))}
              className="w-full"
            />
          </div>

          {/* Epochs */}
          <div className="mb-4">
            <label className="block text-sm font-semibold text-gray-900 mb-2">
              Epochs: <span className="text-purple-600 font-mono">{networkConfig.epochs}</span>
            </label>
            <input
              type="number"
              min="1"
              max="1000"
              value={networkConfig.epochs}
              onChange={(e) => setNetworkConfig(prev => ({ ...prev, epochs: parseInt(e.target.value) || 1 }))}
              className="w-full px-3 py-2 border border-gray-300 rounded text-gray-900 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {/* Train Button */}
          <div className="mb-4">
            {/* Main train/resume button */}
            {!isTraining && !isPaused && (
              <button
                onClick={() => trainNetwork(0)}
                disabled={dataPoints.length === 0}
                className={`w-full px-4 py-2 rounded font-semibold text-sm transition-colors ${
                  dataPoints.length === 0
                    ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                    : 'bg-blue-600 text-white hover:bg-blue-700'
                }`}
              >
                Train Network
              </button>
            )}

            {/* Paused state - show resume button */}
            {isPaused && !isTraining && (
              <div className="space-y-2">
                <div className="text-sm text-center text-amber-600 font-medium">
                  Paused at Epoch {currentEpoch}/{networkConfig.epochs}
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={resumeTraining}
                    className="flex-1 px-4 py-2 rounded font-semibold text-sm bg-green-600 text-white hover:bg-green-700 transition-colors"
                  >
                    Resume
                  </button>
                  <button
                    onClick={stopTraining}
                    className="flex-1 px-4 py-2 rounded font-semibold text-sm bg-red-600 text-white hover:bg-red-700 transition-colors"
                  >
                    Stop
                  </button>
                </div>
              </div>
            )}

            {/* Training state - show pause/stop buttons */}
            {isTraining && (
              <div className="space-y-2">
                <div className="text-sm text-center text-blue-600 font-medium">
                  Training... ({currentEpoch}/{networkConfig.epochs})
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={pauseTraining}
                    className="flex-1 px-4 py-2 rounded font-semibold text-sm bg-amber-500 text-white hover:bg-amber-600 transition-colors"
                  >
                    Pause
                  </button>
                  <button
                    onClick={stopTraining}
                    className="flex-1 px-4 py-2 rounded font-semibold text-sm bg-red-600 text-white hover:bg-red-700 transition-colors"
                  >
                    Stop
                  </button>
                </div>
              </div>
            )}
            {currentLoss !== undefined && (
              <div className="mt-2 text-xs text-gray-600">
                <div>Loss: <span className="font-mono text-red-600">{currentLoss.toFixed(6)}</span></div>
                {lossHistory.length > 0 && (
                  <div className="mt-1">
                    <div className="text-xs text-gray-500 mb-1">Loss History:</div>
                    <div className="h-12 bg-gray-100 rounded relative">
                      <svg className="w-full h-full">
                        {lossHistory.length > 1 && (
                          <polyline
                            fill="none"
                            stroke="#ef4444"
                            strokeWidth="1.5"
                            points={lossHistory.map((loss, i) => {
                              const maxLoss = Math.max(...lossHistory);
                              const minLoss = Math.min(...lossHistory);
                              const range = maxLoss - minLoss || 1;
                              const x = (i / (lossHistory.length - 1)) * 100;
                              const y = 100 - ((loss - minLoss) / range) * 80 - 10;
                              return `${x}%,${y}%`;
                            }).join(' ')}
                          />
                        )}
                      </svg>
                    </div>
                    <div className="flex justify-between text-xs text-gray-400 mt-0.5">
                      <span>Epoch 1</span>
                      <span>Epoch {lossHistory.length}</span>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Data Generators */}
          <div className="mb-4 border-t pt-4">
            <label className="block text-sm font-semibold text-gray-900 mb-2">Data Generators</label>
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={generateLinearData}
                className="px-2 py-2 text-xs bg-green-100 text-green-700 rounded hover:bg-green-200 transition-colors"
                title="Generate linear data (y = 2x + 3)"
              >
                Linear
              </button>
              <button
                onClick={generateNoisyLinearData}
                className="px-2 py-2 text-xs bg-blue-100 text-blue-700 rounded hover:bg-blue-200 transition-colors"
                title="Generate noisy linear data"
              >
                Noisy Linear
              </button>
              <button
                onClick={generateQuadraticData}
                className="px-2 py-2 text-xs bg-purple-100 text-purple-700 rounded hover:bg-purple-200 transition-colors"
                title="Generate quadratic data"
              >
                Quadratic
              </button>
              <button
                onClick={generateSineData}
                className="px-2 py-2 text-xs bg-pink-100 text-pink-700 rounded hover:bg-pink-200 transition-colors"
                title="Generate sine wave data"
              >
                Sine Wave
              </button>
              <button
                onClick={generateRandomScatteredData}
                className="px-2 py-2 text-xs bg-orange-100 text-orange-700 rounded hover:bg-orange-200 transition-colors"
                title="Generate random scattered data"
              >
                Random
              </button>
              <button
                onClick={generateXORData}
                className="px-2 py-2 text-xs bg-indigo-100 text-indigo-700 rounded hover:bg-indigo-200 transition-colors"
                title="Generate XOR-like data for classification"
              >
                XOR Pattern
              </button>
              <button
                onClick={clearAllData}
                disabled={dataPoints.length === 0}
                className="px-2 py-2 text-xs bg-red-100 text-red-700 rounded hover:bg-red-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors col-span-2"
                title="Clear all data points"
              >
                Clear All
              </button>
            </div>
            <div className="mt-2 text-xs font-medium text-gray-700">
              Points: <span className="text-blue-600 font-semibold">{dataPoints.length}</span>
            </div>
          </div>
        </div>

        {/* Center - Network Graph and Data Canvas */}
        <div className="flex-1 flex flex-col gap-3 min-w-0 min-h-0">
          <div className="flex-1 bg-white rounded-lg shadow-sm p-4 min-h-0">
            <h3 className="text-sm font-semibold text-gray-700 mb-2">Network Architecture</h3>
            <div className="h-full">
              <NetworkGraph
                config={networkConfig}
                currentLayer={selectedLayer}
                onNodeClick={handleNodeClick}
              />
            </div>
          </div>
          <div className="flex-1 bg-white rounded-lg shadow-sm p-4 min-h-0 flex flex-col">
            <h3 className="text-sm font-semibold text-gray-700 mb-2">Data Points</h3>
            <div className="flex-1 min-h-0">
              <DataCanvas
                dataPoints={dataPoints}
                onDataPointsChange={setDataPoints}
                mode="regression"
                predictionCurve={predictionCurve}
                xMin={bounds.xMin}
                xMax={bounds.xMax}
                yMin={bounds.yMin}
                yMax={bounds.yMax}
              />
            </div>
          </div>
        </div>

        {/* Right - Matrix View */}
        <div className="w-96 flex-shrink-0">
          <MatrixView
            computation={currentComputation}
            selectedLayer={selectedLayer}
            {...selectedLayerDetails}
          />
        </div>
      </div>

      {/* Debug Controls */}
      <DebugControls
        onNextStep={handleNextStep}
        onPreviousStep={handlePreviousStep}
        onReset={handleReset}
        onAutoStep={handleAutoStep}
        currentStep={currentStep}
        totalSteps={totalSteps}
        phase={currentPhase}
        currentLayer={selectedLayer}
        canGoBack={canGoBack}
        canGoForward={canGoForward}
        isAutoStepping={isAutoStepping}
      />
    </div>
  );
}


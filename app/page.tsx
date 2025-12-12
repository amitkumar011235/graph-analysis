'use client';

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import dynamic from 'next/dynamic';
import DataCanvas from '@/components/DataCanvas';
import ParameterControls from '@/components/ParameterControls';
import { Network } from '@/lib/neural-network';
import { DataPoint, LayerConfig, Mode } from '@/lib/types';

// Dynamically import GraphCalculator to avoid SSR issues
const GraphCalculator = dynamic(() => import('@/components/GraphCalculator'), { ssr: false });

type AppTab = 'neural-network' | 'graph-calculator';

export default function Home() {
  const [activeTab, setActiveTab] = useState<AppTab>('neural-network');
  const [mode, setMode] = useState<Mode>('regression');
  const [dataPoints, setDataPoints] = useState<DataPoint[]>([]);
  const [layers, setLayers] = useState<LayerConfig[]>([
    { neurons: 8, activation: 'relu' },
    { neurons: 4, activation: 'relu' },
    { neurons: 1, activation: 'linear' },
  ]);
  const [learningRate, setLearningRate] = useState(0.01);
  const [epochs, setEpochs] = useState(10);
  const [isTraining, setIsTraining] = useState(false);
  const [autoTrain, setAutoTrain] = useState(false);
  const [currentLoss, setCurrentLoss] = useState<number | undefined>();
  const [predictionCurve, setPredictionCurve] = useState<Array<{ x: number; y: number }>>([]);
  const [decisionBoundary, setDecisionBoundary] = useState<number[][] | undefined>();
  const [network, setNetwork] = useState<Network | null>(null);
  const [isMounted, setIsMounted] = useState(false);
  const [hasTrained, setHasTrained] = useState(false);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [totalEpochs, setTotalEpochs] = useState(0);

  // Client-side mounting check
  useEffect(() => {
    setIsMounted(true);
  }, []);

  // Initialize or recreate network when architecture changes
  useEffect(() => {
    if (!isMounted) return;
    
    try {
      const inputSize = mode === 'regression' ? 1 : 2;
      const lossType = mode === 'regression' ? 'mse' : 'crossentropy';
      const newNetwork = new Network(layers, lossType, inputSize);
      setNetwork(newNetwork);
      setCurrentLoss(undefined);
      setPredictionCurve([]);
      setDecisionBoundary(undefined);
      setHasTrained(false);
      setCurrentEpoch(0);
      setTotalEpochs(0);
    } catch (error) {
      console.error('Error initializing network:', error);
    }
  }, [layers, mode, isMounted]);

  // Convert data points to tensors
  const dataToTensors = useCallback((points: DataPoint[]) => {
    if (points.length === 0) return { x: null, y: null };

    if (mode === 'regression') {
      const x = {
        data: points.map((p) => [p.x]),
        rows: points.length,
        cols: 1,
      };
      const y = {
        data: points.map((p) => [p.y]),
        rows: points.length,
        cols: 1,
      };
      return { x, y };
    } else {
      // Classification: 2D input
      const x = {
        data: points.map((p) => [p.x, p.y]),
        rows: points.length,
        cols: 2,
      };
      const y = {
        data: points.map((p) => [p.label || 0]),
        rows: points.length,
        cols: 1,
      };
      return { x, y };
    }
  }, [mode]);

  // Generate visualization from network predictions
  const generateVisualization = useCallback(() => {
    if (!network || dataPoints.length === 0) return;

    try {
      if (mode === 'regression') {
        // Generate prediction curve
        const xMin = Math.min(...dataPoints.map((p) => p.x)) - 1;
        const xMax = Math.max(...dataPoints.map((p) => p.x)) + 1;
        const resolution = 100;
        const curve: Array<{ x: number; y: number }> = [];

        for (let i = 0; i <= resolution; i++) {
          const x = xMin + (i / resolution) * (xMax - xMin);
          const y = network.predict1D(x);
          // Skip NaN or Infinity values
          if (isFinite(y)) {
            curve.push({ x, y });
          }
        }

        if (curve.length > 0) {
          setPredictionCurve(curve);
        }
      } else {
        // Generate decision boundary
        const xMin = Math.min(...dataPoints.map((p) => p.x)) - 1;
        const xMax = Math.max(...dataPoints.map((p) => p.x)) + 1;
        const yMin = Math.min(...dataPoints.map((p) => p.y)) - 1;
        const yMax = Math.max(...dataPoints.map((p) => p.y)) + 1;
        const resolution = 50;

        const boundary = network.predict2D(xMin, xMax, yMin, yMax, resolution);
        setDecisionBoundary(boundary);
      }
    } catch (error) {
      console.error('Error in generateVisualization:', error);
    }
  }, [network, dataPoints, mode]);

  // Train network - runs epoch by epoch with UI updates
  const trainNetwork = useCallback(async () => {
    if (!network || dataPoints.length === 0) {
      return;
    }

    setIsTraining(true);
    setHasTrained(true);
    setTotalEpochs(epochs);
    setCurrentEpoch(0);
    
    try {
      const { x, y } = dataToTensors(dataPoints);
      if (!x || !y) {
        setIsTraining(false);
        return;
      }

      // Train epoch by epoch for visual feedback
      for (let epoch = 1; epoch <= epochs; epoch++) {
        // Train single epoch
        const lossHistory = network.train(x, y, 1, learningRate);
        
        // Update current epoch and loss
        setCurrentEpoch(epoch);
        if (lossHistory.length > 0) {
          const loss = lossHistory[0];
          // Only update if loss is a valid number
          if (isFinite(loss)) {
            setCurrentLoss(loss);
          }
        }

        // Update visualization after each epoch
        generateVisualization();

        // Small delay to allow UI to render and show progress
        await new Promise((resolve) => setTimeout(resolve, 30));
      }
    } catch (error) {
      console.error('Training error:', error);
    } finally {
      setIsTraining(false);
    }
  }, [network, dataPoints, epochs, learningRate, dataToTensors, generateVisualization]);

  // Auto-train effect
  useEffect(() => {
    if (autoTrain && !isTraining && dataPoints.length > 0 && network) {
      const interval = setInterval(() => {
        trainNetwork();
      }, 1000); // Train every second

      return () => clearInterval(interval);
    }
  }, [autoTrain, isTraining, dataPoints.length, network, trainNetwork]);

  // Update visualization when data changes and we've already trained
  useEffect(() => {
    if (network && dataPoints.length > 0 && hasTrained) {
      generateVisualization();
    }
  }, [network, dataPoints.length, mode, hasTrained, generateVisualization]);

  const handleReset = () => {
    const inputSize = mode === 'regression' ? 1 : 2;
    const lossType = mode === 'regression' ? 'mse' : 'crossentropy';
    const newNetwork = new Network(layers, lossType, inputSize);
    setNetwork(newNetwork);
    setCurrentLoss(undefined);
    setPredictionCurve([]);
    setDecisionBoundary(undefined);
    setHasTrained(false);
    setCurrentEpoch(0);
    setTotalEpochs(0);
  };

  // Generate random data points
  const generateRandomData = useCallback((count: number) => {
    const newPoints: DataPoint[] = [];
    
    if (mode === 'regression') {
      // Generate points following a noisy pattern (sine wave + noise)
      for (let i = 0; i < count; i++) {
        const x = (Math.random() - 0.5) * 8; // Range -4 to 4
        // Create a pattern: sine wave with some noise
        const pattern = Math.sin(x) * 2 + x * 0.3;
        const noise = (Math.random() - 0.5) * 1.5;
        const y = pattern + noise;
        newPoints.push({ x, y });
      }
    } else {
      // Classification: generate clustered points with different classes
      const numClasses = 3;
      // Create cluster centers
      const centers = [
        { x: -2, y: 2 },   // Class 0
        { x: 2, y: 2 },    // Class 1
        { x: 0, y: -2 },   // Class 2
      ];
      
      for (let i = 0; i < count; i++) {
        const label = i % numClasses;
        const center = centers[label];
        // Add Gaussian-like noise around the center
        const x = center.x + (Math.random() - 0.5) * 3;
        const y = center.y + (Math.random() - 0.5) * 3;
        newPoints.push({ x, y, label });
      }
    }
    
    setDataPoints(newPoints);
    setPredictionCurve([]);
    setDecisionBoundary(undefined);
    setHasTrained(false);
    setCurrentEpoch(0);
    setTotalEpochs(0);
  }, [mode]);

  // Clear all data points
  const clearData = useCallback(() => {
    setDataPoints([]);
    setPredictionCurve([]);
    setDecisionBoundary(undefined);
    setHasTrained(false);
    setCurrentEpoch(0);
    setTotalEpochs(0);
  }, []);

  // Calculate canvas bounds - always include origin and ensure minimum visible range
  const bounds = useMemo(() => {
    // Default range centered at origin
    const defaultRange = 5;
    
    if (dataPoints.length === 0) {
      return { xMin: -defaultRange, xMax: defaultRange, yMin: -defaultRange, yMax: defaultRange };
    }

    const xs = dataPoints.map((p) => p.x);
    const ys = dataPoints.map((p) => p.y);
    
    // Calculate data bounds including origin (0,0)
    const dataXMin = Math.min(0, ...xs);
    const dataXMax = Math.max(0, ...xs);
    const dataYMin = Math.min(0, ...ys);
    const dataYMax = Math.max(0, ...ys);
    
    // Add padding (20% of range, minimum 1 unit)
    const xRange = Math.max(dataXMax - dataXMin, 2);
    const yRange = Math.max(dataYMax - dataYMin, 2);
    const padding = Math.max(xRange, yRange) * 0.2;
    
    let xMin = dataXMin - padding;
    let xMax = dataXMax + padding;
    let yMin = dataYMin - padding;
    let yMax = dataYMax + padding;
    
    // Ensure we always have a reasonable minimum range
    const minRange = 4;
    if (xMax - xMin < minRange) {
      const center = (xMin + xMax) / 2;
      xMin = center - minRange / 2;
      xMax = center + minRange / 2;
    }
    if (yMax - yMin < minRange) {
      const center = (yMin + yMax) / 2;
      yMin = center - minRange / 2;
      yMax = center + minRange / 2;
    }
    
    // Make it square (maintain aspect ratio)
    const currentXRange = xMax - xMin;
    const currentYRange = yMax - yMin;
    
    if (currentXRange > currentYRange) {
      const centerY = (yMin + yMax) / 2;
      yMin = centerY - currentXRange / 2;
      yMax = centerY + currentXRange / 2;
    } else {
      const centerX = (xMin + xMax) / 2;
      xMin = centerX - currentYRange / 2;
      xMax = centerX + currentYRange / 2;
    }

    return { xMin, xMax, yMin, yMax };
  }, [dataPoints]);

  if (!isMounted) {
    return (
      <div className="min-h-screen bg-gray-100 p-4 flex items-center justify-center">
        <div className="text-gray-600">Loading...</div>
      </div>
    );
  }

  return (
    <div className="h-screen bg-gray-100 p-2 overflow-hidden flex flex-col">
      {/* Header with Tabs */}
      <div className="flex items-center justify-between mb-2 flex-shrink-0">
        {/* Tab Navigation */}
        <div className="flex items-center gap-1">
          <button
            onClick={() => setActiveTab('neural-network')}
            className={`px-4 py-2 rounded-t-lg font-semibold text-sm transition-colors ${
              activeTab === 'neural-network'
                ? 'bg-white text-blue-600 shadow-sm'
                : 'bg-gray-200 text-gray-600 hover:bg-gray-300'
            }`}
          >
            🧠 Neural Network
          </button>
          <button
            onClick={() => setActiveTab('graph-calculator')}
            className={`px-4 py-2 rounded-t-lg font-semibold text-sm transition-colors ${
              activeTab === 'graph-calculator'
                ? 'bg-white text-blue-600 shadow-sm'
                : 'bg-gray-200 text-gray-600 hover:bg-gray-300'
            }`}
          >
            📈 Graph Calculator
          </button>
        </div>
        
        {/* Training Progress Display (only show for neural network tab) */}
        {activeTab === 'neural-network' && (
          <div className="flex items-center gap-4">
            {(isTraining || hasTrained) && (
              <div className="flex items-center gap-4 bg-white px-4 py-2 rounded-lg shadow-sm border">
                {isTraining ? (
                  <>
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                      <span className="text-sm font-semibold text-blue-600">
                        Epoch {currentEpoch} / {totalEpochs}
                      </span>
                    </div>
                    <div className="w-32 h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-blue-500 transition-all duration-100"
                        style={{ width: `${(currentEpoch / totalEpochs) * 100}%` }}
                      ></div>
                    </div>
                  </>
                ) : (
                  <span className="text-sm text-gray-600">
                    Trained: {totalEpochs} epochs
                  </span>
                )}
                {currentLoss !== undefined && (
                  <div className="flex items-center gap-1">
                    <span className="text-xs text-gray-500">Loss:</span>
                    <span className={`text-sm font-mono font-semibold ${
                      currentLoss < 0.01 ? 'text-green-600' : 
                      currentLoss < 0.1 ? 'text-yellow-600' : 'text-red-600'
                    }`}>
                      {currentLoss.toFixed(6)}
                    </span>
                  </div>
                )}
              </div>
            )}
            <div className="text-xs text-gray-500 hidden md:block">
              Click to add • Drag to move • Right-click to delete
            </div>
          </div>
        )}
      </div>

      {/* Main Content Area - Both tabs always mounted for state preservation */}
      <div className="flex-1 min-h-0 overflow-hidden relative">
        {/* Neural Network Tab */}
        <div 
          className={`absolute inset-0 ${activeTab === 'neural-network' ? 'block' : 'hidden'}`}
        >
          <div className="flex gap-3 h-full">
            {/* Left Sidebar - Parameters (compact) */}
            <div className="w-64 flex-shrink-0 overflow-y-auto">
              <ParameterControls
                mode={mode}
                onModeChange={setMode}
                layers={layers}
                onLayersChange={setLayers}
                learningRate={learningRate}
                onLearningRateChange={setLearningRate}
                epochs={epochs}
                onEpochsChange={setEpochs}
                onTrain={trainNetwork}
                onReset={handleReset}
                isTraining={isTraining}
                autoTrain={autoTrain}
                onAutoTrainChange={setAutoTrain}
                currentLoss={currentLoss}
              />
            </div>

            {/* Center - Canvas (takes all remaining space) */}
            <div className="flex-1 flex flex-col items-center justify-center min-w-0 min-h-0">
              {/* Data Generation Controls */}
              <div className="flex items-center gap-2 mb-2 flex-wrap justify-center">
                <span className="text-xs text-gray-500">Add random points:</span>
                {[10, 50, 100, 200, 500].map((count) => (
                  <button
                    key={count}
                    onClick={() => generateRandomData(count)}
                    disabled={isTraining}
                    className="px-2 py-1 text-xs bg-blue-100 text-blue-700 rounded hover:bg-blue-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    {count}
                  </button>
                ))}
                <button
                  onClick={clearData}
                  disabled={isTraining || dataPoints.length === 0}
                  className="px-2 py-1 text-xs bg-red-100 text-red-700 rounded hover:bg-red-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  Clear All
                </button>
                <span className="text-xs text-gray-400 ml-2">
                  ({dataPoints.length} points)
                </span>
              </div>
              
              <DataCanvas
                dataPoints={dataPoints}
                onDataPointsChange={setDataPoints}
                mode={mode}
                predictionCurve={predictionCurve}
                decisionBoundary={decisionBoundary}
                xMin={bounds.xMin}
                xMax={bounds.xMax}
                yMin={bounds.yMin}
                yMax={bounds.yMax}
              />
              {dataPoints.length === 0 && (
                <div className="mt-2 text-sm text-gray-500 text-center">
                  Click on the canvas to add data points
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Graph Calculator Tab - Always mounted, hidden when not active */}
        <div 
          className={`absolute inset-0 ${activeTab === 'graph-calculator' ? 'block' : 'hidden'}`}
        >
          <GraphCalculator />
        </div>
      </div>
    </div>
  );
}

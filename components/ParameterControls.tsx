'use client';

import React from 'react';
import { LayerConfig, Mode } from '@/lib/types';

interface ParameterControlsProps {
  mode: Mode;
  onModeChange: (mode: Mode) => void;
  layers: LayerConfig[];
  onLayersChange: (layers: LayerConfig[]) => void;
  learningRate: number;
  onLearningRateChange: (lr: number) => void;
  epochs: number;
  onEpochsChange: (epochs: number) => void;
  onTrain: () => void;
  onReset: () => void;
  isTraining: boolean;
  autoTrain: boolean;
  onAutoTrainChange: (auto: boolean) => void;
  currentLoss?: number;
}

export default function ParameterControls({
  mode,
  onModeChange,
  layers,
  onLayersChange,
  learningRate,
  onLearningRateChange,
  epochs,
  onEpochsChange,
  onTrain,
  onReset,
  isTraining,
  autoTrain,
  onAutoTrainChange,
  currentLoss,
}: ParameterControlsProps) {
  const maxLayers = 5;
  const minLayers = 1;

  const addLayer = () => {
    if (layers.length < maxLayers) {
      const lastNeurons = layers.length > 0 ? layers[layers.length - 1].neurons : 8;
      onLayersChange([
        ...layers,
        { neurons: lastNeurons, activation: 'relu' },
      ]);
    }
  };

  const removeLayer = () => {
    if (layers.length > minLayers) {
      onLayersChange(layers.slice(0, -1));
    }
  };

  const updateLayer = (index: number, updates: Partial<LayerConfig>) => {
    const newLayers = [...layers];
    newLayers[index] = { ...newLayers[index], ...updates };
    onLayersChange(newLayers);
  };

  return (
    <div className="w-full bg-white border border-gray-200 rounded-lg p-3 space-y-4 overflow-y-auto">
      <h2 className="text-lg font-bold text-gray-800">Parameters</h2>

      {/* Mode Selection */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Mode
        </label>
        <div className="flex gap-2">
          <button
            onClick={() => onModeChange('regression')}
            className={`flex-1 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              mode === 'regression'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            Regression
          </button>
          <button
            onClick={() => onModeChange('classification')}
            className={`flex-1 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              mode === 'classification'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            Classification
          </button>
        </div>
      </div>

      {/* Network Architecture */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <label className="block text-sm font-medium text-gray-700">
            Layers ({layers.length})
          </label>
          <div className="flex gap-1">
            <button
              onClick={addLayer}
              disabled={layers.length >= maxLayers}
              className="px-2 py-1 text-xs bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed"
            >
              +
            </button>
            <button
              onClick={removeLayer}
              disabled={layers.length <= minLayers}
              className="px-2 py-1 text-xs bg-red-500 text-white rounded hover:bg-red-600 disabled:bg-gray-300 disabled:cursor-not-allowed"
            >
              −
            </button>
          </div>
        </div>

        <div className="space-y-3">
          {layers.map((layer, index) => (
            <div key={index} className="p-3 bg-gray-50 rounded-md border border-gray-200">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-gray-700">
                  Layer {index + 1}
                </span>
              </div>

              {/* Neurons */}
              <div className="mb-2">
                <label className="block text-xs text-gray-600 mb-1">
                  Neurons: {layer.neurons}
                </label>
                <input
                  type="range"
                  min="1"
                  max="32"
                  value={layer.neurons}
                  onChange={(e) =>
                    updateLayer(index, { neurons: parseInt(e.target.value) })
                  }
                  className="w-full"
                />
              </div>

              {/* Activation */}
              <div>
                <label className="block text-xs text-gray-600 mb-1">
                  Activation
                </label>
                <select
                  value={layer.activation}
                  onChange={(e) =>
                    updateLayer(index, {
                      activation: e.target.value as LayerConfig['activation'],
                    })
                  }
                  className="w-full px-2 py-1 text-sm border border-gray-300 rounded-md bg-white"
                >
                  <option value="relu">ReLU</option>
                  <option value="sigmoid">Sigmoid</option>
                  <option value="tanh">Tanh</option>
                  <option value="linear">Linear</option>
                  {mode === 'classification' && index === layers.length - 1 && (
                    <option value="softmax">Softmax</option>
                  )}
                </select>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Hyperparameters */}
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Learning Rate: {learningRate.toFixed(4)}
          </label>
          <input
            type="range"
            min="0.0001"
            max="1"
            step="0.0001"
            value={learningRate}
            onChange={(e) => onLearningRateChange(parseFloat(e.target.value))}
            className="w-full"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>0.0001</span>
            <span>1.0</span>
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Training Epochs: {epochs}
          </label>
          <input
            type="range"
            min="1"
            max="500"
            value={epochs}
            onChange={(e) => onEpochsChange(parseInt(e.target.value))}
            className="w-full"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>1</span>
            <span>500</span>
          </div>
        </div>
      </div>

      {/* Training Controls */}
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            id="autoTrain"
            checked={autoTrain}
            onChange={(e) => onAutoTrainChange(e.target.checked)}
            className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
          />
          <label htmlFor="autoTrain" className="text-sm text-gray-700">
            Auto-train
          </label>
        </div>

        <div className="flex gap-2">
          <button
            onClick={onTrain}
            disabled={isTraining}
            className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
          >
            {isTraining ? 'Training...' : 'Train'}
          </button>
          <button
            onClick={onReset}
            className="px-4 py-2 bg-gray-500 text-white rounded-md hover:bg-gray-600 transition-colors"
          >
            Reset
          </button>
        </div>
      </div>

      {/* Loss Display */}
      {currentLoss !== undefined && (
        <div className="p-3 bg-gray-50 rounded-md border border-gray-200">
          <div className="text-sm text-gray-600 mb-1">Current Loss</div>
          <div className="text-2xl font-bold text-gray-800">
            {currentLoss.toFixed(4)}
          </div>
          <div className="text-xs text-gray-500 mt-1">
            {currentLoss < 0.01 ? '✓ Excellent fit' : currentLoss < 0.1 ? 'Good fit' : 'Keep training'}
          </div>
        </div>
      )}
    </div>
  );
}


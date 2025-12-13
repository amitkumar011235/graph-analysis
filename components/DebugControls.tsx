'use client';

import React from 'react';

interface DebugControlsProps {
  onNextStep: () => void;
  onPreviousStep: () => void;
  onReset: () => void;
  onAutoStep: (enabled: boolean) => void;
  currentStep: number;
  totalSteps: number;
  phase: 'idle' | 'forward' | 'backward' | 'update' | 'completed';
  currentLayer?: number;
  canGoBack: boolean;
  canGoForward: boolean;
  isAutoStepping: boolean;
}

export default function DebugControls({
  onNextStep,
  onPreviousStep,
  onReset,
  onAutoStep,
  currentStep,
  totalSteps,
  phase,
  currentLayer,
  canGoBack,
  canGoForward,
  isAutoStepping,
}: DebugControlsProps) {
  const getPhaseName = () => {
    switch (phase) {
      case 'forward':
        return 'Forward Pass';
      case 'backward':
        return 'Backward Pass';
      case 'update':
        return 'Weight Update';
      case 'completed':
        return 'Completed';
      default:
        return 'Ready';
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm p-4 border-t-2 border-gray-200">
      <div className="flex items-center justify-between gap-4">
        {/* Step Navigation */}
        <div className="flex items-center gap-2">
          <button
            onClick={onPreviousStep}
            disabled={!canGoBack}
            className={`px-4 py-2 rounded font-semibold text-sm transition-colors ${
              canGoBack
                ? 'bg-blue-600 text-white hover:bg-blue-700'
                : 'bg-gray-300 text-gray-500 cursor-not-allowed'
            }`}
          >
            ◀ Previous Step
          </button>
          <button
            onClick={onNextStep}
            disabled={!canGoForward}
            className={`px-4 py-2 rounded font-semibold text-sm transition-colors ${
              canGoForward
                ? 'bg-blue-600 text-white hover:bg-blue-700'
                : 'bg-gray-300 text-gray-500 cursor-not-allowed'
            }`}
          >
            Next Step ▶
          </button>
          <button
            onClick={() => onAutoStep(!isAutoStepping)}
            className={`px-4 py-2 rounded font-semibold text-sm transition-colors ${
              isAutoStepping
                ? 'bg-red-600 text-white hover:bg-red-700'
                : 'bg-green-600 text-white hover:bg-green-700'
            }`}
          >
            {isAutoStepping ? '⏸ Pause' : '▶ Auto Step'}
          </button>
          <button
            onClick={onReset}
            className="px-4 py-2 rounded font-semibold text-sm bg-gray-600 text-white hover:bg-gray-700 transition-colors"
          >
            Reset
          </button>
        </div>

        {/* Status Display */}
        <div className="flex items-center gap-6 text-sm">
          <div className="flex items-center gap-2">
            <span className="text-gray-600">Step:</span>
            <span className="font-mono font-semibold text-blue-600">
              {currentStep} / {totalSteps}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-gray-600">Phase:</span>
            <span className="font-semibold text-purple-600">{getPhaseName()}</span>
          </div>
          {currentLayer !== undefined && (
            <div className="flex items-center gap-2">
              <span className="text-gray-600">Layer:</span>
              <span className="font-semibold text-green-600">{currentLayer}</span>
            </div>
          )}
        </div>
      </div>

      {/* Progress Bar */}
      {totalSteps > 0 && (
        <div className="mt-4">
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${(currentStep / totalSteps) * 100}%` }}
            ></div>
          </div>
        </div>
      )}
    </div>
  );
}


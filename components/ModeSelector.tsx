'use client';

import React from 'react';
import { Mode } from '@/lib/types';

interface ModeSelectorProps {
  mode: Mode;
  onModeChange: (mode: Mode) => void;
}

export default function ModeSelector({ mode, onModeChange }: ModeSelectorProps) {
  return (
    <div className="flex gap-2">
      <button
        onClick={() => onModeChange('regression')}
        className={`px-4 py-2 rounded-md font-medium transition-colors ${
          mode === 'regression'
            ? 'bg-blue-600 text-white'
            : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
        }`}
      >
        Regression
      </button>
      <button
        onClick={() => onModeChange('classification')}
        className={`px-4 py-2 rounded-md font-medium transition-colors ${
          mode === 'classification'
            ? 'bg-blue-600 text-white'
            : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
        }`}
      >
        Classification
      </button>
    </div>
  );
}


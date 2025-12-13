'use client';

import React from 'react';
import { Tensor, ComputationDetail } from '@/lib/types';

interface MatrixViewProps {
  computation?: ComputationDetail;
  selectedLayer?: number;
  weights?: Tensor;
  bias?: number[];
  activations?: Tensor;
  gradients?: {
    weightGrad?: Tensor;
    biasGrad?: number[];
  };
}

export default function MatrixView({
  computation,
  selectedLayer,
  weights,
  bias,
  activations,
  gradients,
}: MatrixViewProps) {
  const formatNumber = (val: number, precision: number = 4): string => {
    if (Math.abs(val) < 1e-10) return '0.0000';
    if (Math.abs(val) >= 1000) return val.toExponential(2);
    return val.toFixed(precision);
  };

  const renderTensor = (tensor: Tensor, title: string, maxRows: number = 10, maxCols: number = 10) => {
    const displayRows = Math.min(tensor.rows, maxRows);
    const displayCols = Math.min(tensor.cols, maxCols);
    const hasMoreRows = tensor.rows > maxRows;
    const hasMoreCols = tensor.cols > maxCols;

    return (
      <div className="mb-4">
        <div className="text-sm font-semibold text-gray-700 mb-2">
          {title} ({tensor.rows}×{tensor.cols})
        </div>
        <div className="overflow-auto max-h-64 border border-gray-300 rounded">
          <table className="min-w-full text-xs">
            <thead>
              <tr>
                <th className="px-2 py-1 bg-gray-100 border-b border-gray-300"></th>
                {Array.from({ length: displayCols }).map((_, j) => (
                  <th key={j} className="px-2 py-1 bg-gray-100 border-b border-gray-300 text-center">
                    {j}
                  </th>
                ))}
                {hasMoreCols && (
                  <th className="px-2 py-1 bg-gray-100 border-b border-gray-300 text-center">...</th>
                )}
              </tr>
            </thead>
            <tbody>
              {Array.from({ length: displayRows }).map((_, i) => (
                <tr key={i}>
                  <td className="px-2 py-1 bg-gray-100 border-r border-gray-300 font-semibold">{i}</td>
                  {Array.from({ length: displayCols }).map((_, j) => {
                    const val = tensor.data[i][j];
                    const isPositive = val >= 0;
                    return (
                      <td
                        key={j}
                        className={`px-2 py-1 text-right border border-gray-200 ${
                          isPositive ? 'text-green-700' : 'text-red-700'
                        }`}
                      >
                        {formatNumber(val)}
                      </td>
                    );
                  })}
                  {hasMoreCols && (
                    <td className="px-2 py-1 text-center border border-gray-200">...</td>
                  )}
                </tr>
              ))}
              {hasMoreRows && (
                <tr>
                  <td className="px-2 py-1 bg-gray-100 border-r border-gray-300">...</td>
                  {Array.from({ length: displayCols + (hasMoreCols ? 1 : 0) }).map((_, j) => (
                    <td key={j} className="px-2 py-1 text-center border border-gray-200">...</td>
                  ))}
                </tr>
              )}
            </tbody>
          </table>
        </div>
        {(hasMoreRows || hasMoreCols) && (
          <div className="text-xs text-gray-500 mt-1">
            Showing {displayRows}×{displayCols} of {tensor.rows}×{tensor.cols}
          </div>
        )}
      </div>
    );
  };

  const renderArray = (arr: number[], title: string) => {
    const maxDisplay = 20;
    const displayArr = arr.slice(0, maxDisplay);
    const hasMore = arr.length > maxDisplay;

    return (
      <div className="mb-4">
        <div className="text-sm font-semibold text-gray-700 mb-2">
          {title} ({arr.length})
        </div>
        <div className="overflow-auto max-h-48 border border-gray-300 rounded p-2">
          <div className="flex flex-wrap gap-2">
            {displayArr.map((val, i) => {
              const isPositive = val >= 0;
              return (
                <span
                  key={i}
                  className={`px-2 py-1 rounded text-xs font-mono ${
                    isPositive ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'
                  }`}
                >
                  [{i}]: {formatNumber(val)}
                </span>
              );
            })}
            {hasMore && (
              <span className="px-2 py-1 text-xs text-gray-500">... ({arr.length - maxDisplay} more)</span>
            )}
          </div>
        </div>
      </div>
    );
  };

  const renderScalar = (val: number, title: string) => {
    return (
      <div className="mb-4">
        <div className="text-sm font-semibold text-gray-700 mb-2">{title}</div>
        <div className="text-lg font-mono p-2 bg-gray-50 border border-gray-300 rounded">
          {formatNumber(val, 6)}
        </div>
      </div>
    );
  };

  return (
    <div className="h-full overflow-y-auto p-4 bg-white rounded-lg shadow-sm">
      <h3 className="text-lg font-bold text-gray-800 mb-4">Computation Details</h3>

      {computation && (
        <div className="mb-6">
          <div className="text-sm font-semibold text-gray-700 mb-2">Operation</div>
          <div className="text-sm text-gray-600 mb-2">{computation.description}</div>
          <div className="text-sm font-mono bg-gray-50 p-3 rounded border border-gray-200 whitespace-pre-wrap mb-4">
            {computation.formula}
          </div>

          <div className="text-sm font-semibold text-gray-700 mb-2">Inputs</div>
          <div className="space-y-2 mb-4">
            {computation.inputs.map((input, i) => (
              <div key={i}>
                {input.value instanceof Object && 'rows' in input.value ? (
                  renderTensor(input.value as Tensor, input.name)
                ) : Array.isArray(input.value) ? (
                  renderArray(input.value as number[], input.name)
                ) : (
                  renderScalar(input.value as number, input.name)
                )}
              </div>
            ))}
          </div>

          <div className="text-sm font-semibold text-gray-700 mb-2">Output</div>
          <div>
            {computation.output instanceof Object && 'rows' in computation.output ? (
              renderTensor(computation.output as Tensor, 'Output')
            ) : Array.isArray(computation.output) ? (
              renderArray(computation.output as number[], 'Output')
            ) : (
              renderScalar(computation.output as number, 'Output')
            )}
          </div>
        </div>
      )}

      {selectedLayer !== undefined && weights && (
        <div className="border-t pt-4 mt-4">
          <h4 className="text-sm font-semibold text-gray-700 mb-3">Layer {selectedLayer} Details</h4>
          {renderTensor(weights, 'Weights')}
          {bias && renderArray(bias, 'Bias')}
          {gradients?.weightGrad && renderTensor(gradients.weightGrad, 'Weight Gradients')}
          {gradients?.biasGrad && renderArray(gradients.biasGrad, 'Bias Gradients')}
          {activations && renderTensor(activations, 'Activations')}
        </div>
      )}

      {!computation && selectedLayer === undefined && (
        <div className="text-gray-500 text-sm text-center py-8">
          Click on a network node or step through computations to see details
        </div>
      )}
    </div>
  );
}


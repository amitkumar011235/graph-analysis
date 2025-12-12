'use client';

import React from 'react';
import {
  LineChart,
  Line,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { DataPoint } from '@/lib/types';

interface ChartProps {
  dataPoints: DataPoint[];
  predictionCurve?: Array<{ x: number; y: number }>;
  mode: 'regression' | 'classification';
}

export default function Chart({ dataPoints, predictionCurve, mode }: ChartProps) {
  if (mode === 'regression') {
    // Prepare scatter data
    const scatterData = dataPoints.map((point) => ({
      x: point.x,
      y: point.y,
    }));

    // Combine prediction curve with data points
    const allData = predictionCurve
      ? [...predictionCurve.map((p) => ({ x: p.x, y: p.y, type: 'prediction' })), ...scatterData.map((p) => ({ ...p, type: 'data' }))]
      : scatterData.map((p) => ({ ...p, type: 'data' }));

    return (
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={predictionCurve || []}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="x" />
          <YAxis dataKey="y" />
          <Tooltip />
          <Line
            type="monotone"
            dataKey="y"
            stroke="#3b82f6"
            strokeWidth={2}
            dot={false}
          />
          <Scatter data={scatterData} fill="#ef4444">
            {scatterData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill="#ef4444" />
            ))}
          </Scatter>
        </LineChart>
      </ResponsiveContainer>
    );
  }

  // Classification mode - show scatter with different colors
  const colors = ['#ef4444', '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6'];
  const scatterData = dataPoints.map((point) => ({
    x: point.x,
    y: point.y,
    label: point.label || 0,
  }));

  return (
    <ResponsiveContainer width="100%" height={400}>
      <ScatterChart>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="x" />
        <YAxis dataKey="y" />
        <Tooltip />
        <Scatter data={scatterData} fill="#8884d8">
          {scatterData.map((entry, index) => (
            <Cell
              key={`cell-${index}`}
              fill={colors[entry.label] || colors[0]}
            />
          ))}
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
  );
}


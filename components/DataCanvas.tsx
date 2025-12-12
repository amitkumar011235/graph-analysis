'use client';

import React, { useRef, useEffect, useState, useCallback } from 'react';
import { DataPoint } from '@/lib/types';

interface DataCanvasProps {
  dataPoints: DataPoint[];
  onDataPointsChange: (points: DataPoint[]) => void;
  mode: 'regression' | 'classification';
  predictionCurve?: Array<{ x: number; y: number }>;
  decisionBoundary?: number[][];
  xMin?: number;
  xMax?: number;
  yMin?: number;
  yMax?: number;
}

export default function DataCanvas({
  dataPoints,
  onDataPointsChange,
  mode,
  predictionCurve,
  decisionBoundary,
  xMin = -5,
  xMax = 5,
  yMin = -5,
  yMax = 5,
}: DataCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDragging, setIsDragging] = useState<number | null>(null);
  const [selectedClass, setSelectedClass] = useState(0);

  // Responsive canvas size - will be scaled via CSS
  const width = 700;
  const height = 700;
  const padding = 60;

  // Convert data coordinates to canvas coordinates
  const dataToCanvas = useCallback((x: number, y: number) => {
    const canvasX = ((x - xMin) / (xMax - xMin)) * (width - 2 * padding) + padding;
    const canvasY = height - padding - ((y - yMin) / (yMax - yMin)) * (height - 2 * padding);
    return { x: canvasX, y: canvasY };
  }, [xMin, xMax, yMin, yMax, width, height, padding]);

  // Convert canvas coordinates to data coordinates
  const canvasToData = useCallback((canvasX: number, canvasY: number) => {
    // Clamp coordinates to valid canvas area
    const clampedX = Math.max(padding, Math.min(canvasX, width - padding));
    const clampedY = Math.max(padding, Math.min(canvasY, height - padding));
    
    const x = ((clampedX - padding) / (width - 2 * padding)) * (xMax - xMin) + xMin;
    const y = yMax - ((clampedY - padding) / (height - 2 * padding)) * (yMax - yMin);
    return { x, y };
  }, [xMin, xMax, yMin, yMax, width, height, padding]);

  // Find closest point to click (in canvas pixel coordinates)
  const findClosestPoint = useCallback((canvasX: number, canvasY: number, threshold: number = 15) => {
    let closestIndex = -1;
    let minDist = threshold;

    dataPoints.forEach((point, index) => {
      const canvasPos = dataToCanvas(point.x, point.y);
      const dist = Math.sqrt(
        Math.pow(canvasX - canvasPos.x, 2) + Math.pow(canvasY - canvasPos.y, 2)
      );
      if (dist < minDist) {
        minDist = dist;
        closestIndex = index;
      }
    });

    return closestIndex;
  }, [dataPoints, dataToCanvas]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw grid
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
      const x = padding + (i / 10) * (width - 2 * padding);
      const y = padding + (i / 10) * (height - 2 * padding);
      
      // Vertical lines
      ctx.beginPath();
      ctx.moveTo(x, padding);
      ctx.lineTo(x, height - padding);
      ctx.stroke();
      
      // Horizontal lines
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();
    }

    // Draw plot border
    ctx.strokeStyle = '#9ca3af';
    ctx.lineWidth = 1;
    ctx.strokeRect(padding, padding, width - 2 * padding, height - 2 * padding);

    // Calculate origin position (where x=0 and y=0 lines would be)
    const origin = dataToCanvas(0, 0);
    const originInViewX = xMin <= 0 && xMax >= 0;
    const originInViewY = yMin <= 0 && yMax >= 0;
    
    // Draw axes through origin if origin is in view
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 2;
    
    // X axis (horizontal line at y=0)
    if (originInViewY) {
      ctx.beginPath();
      ctx.moveTo(padding, origin.y);
      ctx.lineTo(width - padding, origin.y);
      ctx.stroke();
      
      // Arrow at right end
      ctx.fillStyle = '#374151';
      ctx.beginPath();
      ctx.moveTo(width - padding, origin.y);
      ctx.lineTo(width - padding - 8, origin.y - 4);
      ctx.lineTo(width - padding - 8, origin.y + 4);
      ctx.closePath();
      ctx.fill();
    }
    
    // Y axis (vertical line at x=0)
    if (originInViewX) {
      ctx.beginPath();
      ctx.moveTo(origin.x, padding);
      ctx.lineTo(origin.x, height - padding);
      ctx.stroke();
      
      // Arrow at top
      ctx.fillStyle = '#374151';
      ctx.beginPath();
      ctx.moveTo(origin.x, padding);
      ctx.lineTo(origin.x - 4, padding + 8);
      ctx.lineTo(origin.x + 4, padding + 8);
      ctx.closePath();
      ctx.fill();
    }

    // Draw axis labels at edges
    ctx.fillStyle = '#6b7280';
    ctx.font = '12px sans-serif';
    
    // X axis labels (bottom edge)
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(`${xMin.toFixed(1)}`, padding, height - padding + 8);
    ctx.fillText(`${xMax.toFixed(1)}`, width - padding, height - padding + 8);
    if (originInViewX && Math.abs(origin.x - padding) > 30 && Math.abs(origin.x - (width - padding)) > 30) {
      ctx.fillText('0', origin.x, height - padding + 8);
    }
    
    // Y axis labels (left edge)
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(`${yMin.toFixed(1)}`, padding - 8, height - padding);
    ctx.fillText(`${yMax.toFixed(1)}`, padding - 8, padding);
    if (originInViewY && Math.abs(origin.y - padding) > 20 && Math.abs(origin.y - (height - padding)) > 20) {
      ctx.fillText('0', padding - 8, origin.y);
    }

    // Draw decision boundary (classification mode)
    if (mode === 'classification' && decisionBoundary) {
      const resolution = decisionBoundary.length;
      const cellWidth = (width - 2 * padding) / resolution;
      const cellHeight = (height - 2 * padding) / resolution;

      for (let i = 0; i < resolution; i++) {
        for (let j = 0; j < resolution; j++) {
          const value = decisionBoundary[i][j];
          const alpha = Math.abs(value);
          const color = value > 0 ? `rgba(59, 130, 246, ${alpha})` : `rgba(239, 68, 68, ${alpha})`;
          
          ctx.fillStyle = color;
          ctx.fillRect(
            padding + j * cellWidth,
            padding + i * cellHeight,
            cellWidth,
            cellHeight
          );
        }
      }
    }

    // Draw prediction curve (regression mode)
    if (mode === 'regression' && predictionCurve && predictionCurve.length > 0) {
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 2;
      ctx.beginPath();
      
      const first = dataToCanvas(predictionCurve[0].x, predictionCurve[0].y);
      ctx.moveTo(first.x, first.y);
      
      for (let i = 1; i < predictionCurve.length; i++) {
        const point = dataToCanvas(predictionCurve[i].x, predictionCurve[i].y);
        ctx.lineTo(point.x, point.y);
      }
      
      ctx.stroke();
    }

    // Draw data points
    dataPoints.forEach((point, index) => {
      const canvasPos = dataToCanvas(point.x, point.y);
      const radius = 8;
      
      // Different colors for classification
      if (mode === 'classification') {
        const colors = ['#ef4444', '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6'];
        ctx.fillStyle = colors[point.label || 0] || colors[0];
      } else {
        ctx.fillStyle = '#3b82f6';
      }
      
      ctx.beginPath();
      ctx.arc(canvasPos.x, canvasPos.y, radius, 0, 2 * Math.PI);
      ctx.fill();
      
      // Draw border
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();
    });
  }, [dataPoints, mode, predictionCurve, decisionBoundary, dataToCanvas, xMin, xMax, yMin, yMax]);

  useEffect(() => {
    draw();
  }, [draw]);

  // Get canvas coordinates accounting for CSS scaling
  const getCanvasCoords = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    
    const rect = canvas.getBoundingClientRect();
    // Scale mouse coordinates to internal canvas coordinates
    const scaleX = width / rect.width;
    const scaleY = height / rect.height;
    
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    
    return { x, y };
  }, [width, height]);

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const coords = getCanvasCoords(e);
    if (!coords) return;

    const { x, y } = coords;

    // Check if clicking on existing point
    const pointIndex = findClosestPoint(x, y);
    
    if (pointIndex >= 0) {
      setIsDragging(pointIndex);
    } else if (e.button === 0) {
      // Left click: add new point
      const data = canvasToData(x, y);
      const newPoint: DataPoint = {
        x: data.x,
        y: data.y,
        label: mode === 'classification' ? selectedClass : undefined,
      };
      onDataPointsChange([...dataPoints, newPoint]);
    } else if (e.button === 2) {
      // Right click: delete point
      if (pointIndex >= 0) {
        const newPoints = dataPoints.filter((_, i) => i !== pointIndex);
        onDataPointsChange(newPoints);
      }
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isDragging !== null) {
      const coords = getCanvasCoords(e);
      if (!coords) return;

      const { x, y } = coords;
      const data = canvasToData(x, y);

      const newPoints = [...dataPoints];
      newPoints[isDragging] = {
        ...newPoints[isDragging],
        x: data.x,
        y: data.y,
      };
      onDataPointsChange(newPoints);
    }
  };

  const handleMouseUp = () => {
    setIsDragging(null);
  };

  return (
    <div className="flex flex-col items-center justify-center w-full h-full gap-1">
      {mode === 'classification' && (
        <div className="flex gap-2 items-center">
          <span className="text-sm text-gray-600">Class:</span>
          {[0, 1, 2, 3, 4].map((cls) => {
            const colors = ['#ef4444', '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6'];
            return (
              <button
                key={cls}
                onClick={() => setSelectedClass(cls)}
                className={`w-8 h-8 rounded-full border-2 ${
                  selectedClass === cls ? 'border-gray-900' : 'border-gray-300'
                }`}
                style={{ backgroundColor: colors[cls] }}
                title={`Class ${cls}`}
              />
            );
          })}
        </div>
      )}
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onContextMenu={(e) => e.preventDefault()}
        className="border border-gray-300 rounded-lg cursor-crosshair bg-white shadow-md"
        style={{ 
          touchAction: 'none', 
          width: '100%',
          height: '100%',
          maxWidth: 'min(calc(100vh - 120px), 100%)',
          maxHeight: 'calc(100vh - 120px)',
          aspectRatio: '1',
          objectFit: 'contain'
        }}
      />
      <p className="text-xs text-gray-400 text-center">
        Left click: add • Drag: move • Right click: delete
      </p>
    </div>
  );
}


'use client';

import React, { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import {
  DataPoint,
  normalEquation,
  computeMSE,
  computeMAE,
  computeHuber,
  computeR2,
  computeLoss,
  gradientDescentStep,
  computeLossLandscape,
  LossType,
} from '@/lib/linear-regression';
import LessonSidebar from './LessonSidebar';

interface RegressionState {
  mode: 'manual' | 'auto-fit' | 'gradient-descent';
  m: number;
  b: number;
  learningRate: number;
  isAnimating: boolean;
  stepCount: number;
  lossHistory: number[];
  trainTestSplit: number;
  lossType: LossType;
  showLossLandscape: boolean;
  showResiduals: boolean;
  showSquaredErrors: boolean;
  showGradients: boolean;
  activeLesson: string | null;
}

export default function LinearRegressionVisualizer() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  // Data points
  const [dataPoints, setDataPoints] = useState<DataPoint[]>([]);
  const [isDragging, setIsDragging] = useState<number | null>(null);
  
  // Regression state
  const [mode, setMode] = useState<'manual' | 'auto-fit' | 'gradient-descent'>('manual');
  const [m, setM] = useState(1.0);
  const [b, setB] = useState(0.0);
  const [learningRate, setLearningRate] = useState(0.1);
  const [isAnimating, setIsAnimating] = useState(false);
  const [stepCount, setStepCount] = useState(0);
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [trainTestSplit, setTrainTestSplit] = useState(0.7);
  const [lossType, setLossType] = useState<LossType>('mse');
  const [showLossLandscape, setShowLossLandscape] = useState(false);
  const [showResiduals, setShowResiduals] = useState(true);
  const [showSquaredErrors, setShowSquaredErrors] = useState(false);
  const [showGradients, setShowGradients] = useState(false);
  const [activeLesson, setActiveLesson] = useState<string | null>(null);
  
  // Gradient info for visualization
  const [gradients, setGradients] = useState<{ dm: number; db: number } | null>(null);
  
  // Loss landscape plot ref
  const lossLandscapeRef = useRef<HTMLDivElement>(null);
  
  // View bounds
  const [xMin, setXMin] = useState(-10);
  const [xMax, setXMax] = useState(10);
  const [yMin, setYMin] = useState(-10);
  const [yMax, setYMax] = useState(10);

  // Canvas dimensions
  const width = 700;
  const height = 700;
  const padding = 60;

  // Compute bounds from data points
  const bounds = useMemo(() => {
    if (dataPoints.length === 0) {
      return { xMin: -10, xMax: 10, yMin: -10, yMax: 10 };
    }

    let xMinVal = Math.min(...dataPoints.map(p => p.x));
    let xMaxVal = Math.max(...dataPoints.map(p => p.x));
    let yMinVal = Math.min(...dataPoints.map(p => p.y));
    let yMaxVal = Math.max(...dataPoints.map(p => p.y));

    // Add padding
    const xRange = xMaxVal - xMinVal;
    const yRange = yMaxVal - yMinVal;
    const paddingFactor = 0.2;

    if (xRange < 0.001) {
      xMinVal -= 5;
      xMaxVal += 5;
    } else {
      const xPadding = xRange * paddingFactor;
      xMinVal -= xPadding;
      xMaxVal += xPadding;
    }

    if (yRange < 0.001) {
      yMinVal -= 5;
      yMaxVal += 5;
    } else {
      const yPadding = yRange * paddingFactor;
      yMinVal -= yPadding;
      yMaxVal += yPadding;
    }

    // Ensure origin is visible if possible
    if (xMinVal > 0) xMinVal = 0;
    if (xMaxVal < 0) xMaxVal = 0;
    if (yMinVal > 0) yMinVal = 0;
    if (yMaxVal < 0) yMaxVal = 0;

    return { xMin: xMinVal, xMax: xMaxVal, yMin: yMinVal, yMax: yMaxVal };
  }, [dataPoints]);

  // Split points into train/test
  const { trainPoints, testPoints } = useMemo(() => {
    if (dataPoints.length === 0) {
      return { trainPoints: [], testPoints: [] };
    }

    const shuffled = [...dataPoints].sort(() => Math.random() - 0.5);
    const splitIndex = Math.floor(shuffled.length * trainTestSplit);
    return {
      trainPoints: shuffled.slice(0, splitIndex),
      testPoints: shuffled.slice(splitIndex),
    };
  }, [dataPoints, trainTestSplit]);

  // Compute current metrics
  const metrics = useMemo(() => {
    const points = trainPoints.length > 0 ? trainPoints : dataPoints;
    if (points.length === 0) {
      return { mse: 0, mae: 0, r2: 0, currentLoss: 0 };
    }

    const mse = computeMSE(points, m, b);
    const mae = computeMAE(points, m, b);
    const r2 = computeR2(points, m, b);
    const currentLoss = computeLoss(points, m, b, lossType);

    return { mse, mae, r2, currentLoss };
  }, [trainPoints, dataPoints, m, b, lossType]);

  // Auto-fit when in auto-fit mode and points change
  useEffect(() => {
    if (mode === 'auto-fit' && trainPoints.length >= 2) {
      const { m: newM, b: newB } = normalEquation(trainPoints);
      setM(newM);
      setB(newB);
      setStepCount(0);
      setLossHistory([]);
    }
  }, [mode, trainPoints]);

  // Coordinate conversion functions
  const dataToCanvas = useCallback(
    (x: number, y: number) => {
      const canvasX = ((x - bounds.xMin) / (bounds.xMax - bounds.xMin)) * (width - 2 * padding) + padding;
      const canvasY = height - padding - ((y - bounds.yMin) / (bounds.yMax - bounds.yMin)) * (height - 2 * padding);
      return { x: canvasX, y: canvasY };
    },
    [bounds, width, height, padding]
  );

  const canvasToData = useCallback(
    (canvasX: number, canvasY: number) => {
      const clampedX = Math.max(padding, Math.min(canvasX, width - padding));
      const clampedY = Math.max(padding, Math.min(canvasY, height - padding));

      const x = ((clampedX - padding) / (width - 2 * padding)) * (bounds.xMax - bounds.xMin) + bounds.xMin;
      const y = bounds.yMax - ((clampedY - padding) / (height - 2 * padding)) * (bounds.yMax - bounds.yMin);
      return { x, y };
    },
    [bounds, width, height, padding]
  );

  // Find closest point
  const findClosestPoint = useCallback(
    (canvasX: number, canvasY: number, threshold: number = 10) => {
      let closestIndex = -1;
      let minDist = threshold;

      for (let i = 0; i < dataPoints.length; i++) {
        const { x: cx, y: cy } = dataToCanvas(dataPoints[i].x, dataPoints[i].y);
        const dist = Math.sqrt((canvasX - cx) ** 2 + (canvasY - cy) ** 2);
        if (dist < minDist) {
          minDist = dist;
          closestIndex = i;
        }
      }

      return closestIndex;
    },
    [dataPoints, dataToCanvas]
  );

  // Get canvas coordinates from mouse event
  const getCanvasCoords = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return null;

      const rect = canvas.getBoundingClientRect();
      const scaleX = width / rect.width;
      const scaleY = height / rect.height;

      const x = (e.clientX - rect.left) * scaleX;
      const y = (e.clientY - rect.top) * scaleY;

      return { x, y };
    },
    [width, height]
  );

  // Mouse handlers
  const handleMouseDown = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const coords = getCanvasCoords(e);
      if (!coords) return;

      const { x, y } = coords;
      const pointIndex = findClosestPoint(x, y);

      if (pointIndex >= 0) {
        setIsDragging(pointIndex);
      } else if (e.button === 0) {
        // Left click: add new point
        const data = canvasToData(x, y);
        const newPoint: DataPoint = {
          x: data.x,
          y: data.y,
          isTrain: true,
        };
        setDataPoints([...dataPoints, newPoint]);
      } else if (e.button === 2) {
        // Right click: delete point
        if (pointIndex >= 0) {
          setDataPoints(dataPoints.filter((_, i) => i !== pointIndex));
        }
      }
    },
    [getCanvasCoords, findClosestPoint, canvasToData, dataPoints]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
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
        setDataPoints(newPoints);
      }
    },
    [isDragging, getCanvasCoords, canvasToData, dataPoints]
  );

  const handleMouseUp = useCallback(() => {
    setIsDragging(null);
  }, []);

  // Gradient descent step
  const performGradientDescentStep = useCallback(() => {
    if (trainPoints.length < 1 || mode !== 'gradient-descent') return;

    const result = gradientDescentStep(trainPoints, m, b, learningRate, lossType);
    setM(result.m);
    setB(result.b);
    setGradients(result.gradients);
    setStepCount(prev => prev + 1);

    // Update loss history
    const loss = computeLoss(trainPoints, result.m, result.b, lossType);
    setLossHistory(prev => [...prev.slice(-99), loss]);
  }, [trainPoints, m, b, learningRate, lossType, mode]);

  // Animation loop for gradient descent
  useEffect(() => {
    if (!isAnimating || mode !== 'gradient-descent') return;

    const interval = setInterval(() => {
      performGradientDescentStep();
    }, 50); // ~20 steps per second

    return () => clearInterval(interval);
  }, [isAnimating, mode, performGradientDescentStep]);

  // Draw function
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

    const gridStep = (bounds.xMax - bounds.xMin) / 20;
    for (let x = Math.ceil(bounds.xMin / gridStep) * gridStep; x <= bounds.xMax; x += gridStep) {
      const { x: cx } = dataToCanvas(x, 0);
      ctx.beginPath();
      ctx.moveTo(cx, padding);
      ctx.lineTo(cx, height - padding);
      ctx.stroke();
    }

    const yGridStep = (bounds.yMax - bounds.yMin) / 20;
    for (let y = Math.ceil(bounds.yMin / yGridStep) * yGridStep; y <= bounds.yMax; y += yGridStep) {
      const { y: cy } = dataToCanvas(0, y);
      ctx.beginPath();
      ctx.moveTo(padding, cy);
      ctx.lineTo(width - padding, cy);
      ctx.stroke();
    }

    // Draw axes
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 2;

    const xAxisY = dataToCanvas(0, 0).y;
    if (xAxisY >= padding && xAxisY <= height - padding) {
      ctx.beginPath();
      ctx.moveTo(padding, xAxisY);
      ctx.lineTo(width - padding, xAxisY);
      ctx.stroke();

      // Arrow
      ctx.beginPath();
      ctx.moveTo(width - padding - 10, xAxisY - 5);
      ctx.lineTo(width - padding, xAxisY);
      ctx.lineTo(width - padding - 10, xAxisY + 5);
      ctx.fill();
    }

    const yAxisX = dataToCanvas(0, 0).x;
    if (yAxisX >= padding && yAxisX <= width - padding) {
      ctx.beginPath();
      ctx.moveTo(yAxisX, height - padding);
      ctx.lineTo(yAxisX, padding);
      ctx.stroke();

      // Arrow
      ctx.beginPath();
      ctx.moveTo(yAxisX - 5, padding + 10);
      ctx.lineTo(yAxisX, padding);
      ctx.lineTo(yAxisX + 5, padding + 10);
      ctx.fill();
    }

    // Draw regression line
    if (dataPoints.length > 0) {
      const x1 = bounds.xMin;
      const x2 = bounds.xMax;
      const y1 = m * x1 + b;
      const y2 = m * x2 + b;

      const p1 = dataToCanvas(x1, y1);
      const p2 = dataToCanvas(x2, y2);

      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();

      // Draw equation label
      ctx.fillStyle = '#3b82f6';
      ctx.font = '14px sans-serif';
      ctx.fillText(`y = ${m.toFixed(2)}x + ${b.toFixed(2)}`, p2.x - 100, p2.y - 10);
    }

    // Draw data points and residuals
    for (let i = 0; i < dataPoints.length; i++) {
      const point = dataPoints[i];
      const isTrain = trainPoints.includes(point);
      const { x: cx, y: cy } = dataToCanvas(point.x, point.y);

      // Draw residual line
      if (showResiduals && dataPoints.length > 0) {
        const predY = m * point.x + b;
        const { y: predCy } = dataToCanvas(point.x, predY);

        ctx.strokeStyle = isTrain ? 'rgba(239, 68, 68, 0.4)' : 'rgba(245, 158, 11, 0.4)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx, predCy);
        ctx.stroke();

        // Draw squared error block
        if (showSquaredErrors) {
          const error = Math.abs(predY - point.y);
          const blockSize = Math.abs(cy - predCy);
          ctx.fillStyle = isTrain ? 'rgba(239, 68, 68, 0.2)' : 'rgba(245, 158, 11, 0.2)';
          ctx.fillRect(cx - blockSize / 2, Math.min(cy, predCy), blockSize, blockSize);
        }
      }

      // Draw point
      ctx.fillStyle = isTrain ? '#ef4444' : '#f59e0b';
      ctx.beginPath();
      ctx.arc(cx, cy, 5, 0, 2 * Math.PI);
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Draw gradient arrows if in GD mode and gradients available
    if (showGradients && mode === 'gradient-descent' && gradients && dataPoints.length > 0) {
      const centerX = (bounds.xMin + bounds.xMax) / 2;
      const centerY = (bounds.yMin + bounds.yMax) / 2;
      const { x: centerCx, y: centerCy } = dataToCanvas(centerX, centerY);

      const scale = 50;
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 2;
      ctx.fillStyle = '#10b981';

      // m gradient arrow
      ctx.beginPath();
      ctx.moveTo(centerCx - 100, centerCy);
      ctx.lineTo(centerCx - 100 + gradients.dm * scale, centerCy);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(centerCx - 100 + gradients.dm * scale - 5, centerCy - 5);
      ctx.lineTo(centerCx - 100 + gradients.dm * scale, centerCy);
      ctx.lineTo(centerCx - 100 + gradients.dm * scale - 5, centerCy + 5);
      ctx.fill();

      // b gradient arrow
      ctx.beginPath();
      ctx.moveTo(centerCx - 100, centerCy + 30);
      ctx.lineTo(centerCx - 100, centerCy + 30 + gradients.db * scale);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(centerCx - 100 - 5, centerCy + 30 + gradients.db * scale - 5);
      ctx.lineTo(centerCx - 100, centerCy + 30 + gradients.db * scale);
      ctx.lineTo(centerCx - 100 + 5, centerCy + 30 + gradients.db * scale - 5);
      ctx.fill();

      ctx.fillStyle = '#10b981';
      ctx.font = '12px sans-serif';
      ctx.fillText(`dm: ${gradients.dm.toFixed(3)}`, centerCx - 150, centerCy - 5);
      ctx.fillText(`db: ${gradients.db.toFixed(3)}`, centerCx - 150, centerCy + 55);
    }
  }, [
    bounds,
    dataToCanvas,
    dataPoints,
    trainPoints,
    m,
    b,
    showResiduals,
    showSquaredErrors,
    showGradients,
    mode,
    gradients,
  ]);

  useEffect(() => {
    draw();
  }, [draw]);

  // Loss landscape 3D plot
  useEffect(() => {
    if (!showLossLandscape || !lossLandscapeRef.current || trainPoints.length < 2) {
      return;
    }

    // Dynamic import to avoid SSR
    import('plotly.js-dist-min').then((PlotlyModule: any) => {
      const Plotly = PlotlyModule.default || PlotlyModule;
      
      // Compute loss landscape around current m, b
      const mRange: [number, number] = [m - 2, m + 2];
      const bRange: [number, number] = [b - 2, b + 2];
      const landscape = computeLossLandscape(trainPoints, mRange, bRange, 40, lossType);

      // Current loss
      const currentLoss = computeLoss(trainPoints, m, b, lossType);

      const data: any[] = [
        {
          type: 'surface',
          x: landscape.m,
          y: landscape.b,
          z: landscape.loss,
          colorscale: 'Viridis',
          showscale: true,
          colorbar: { title: 'Loss' },
        },
        {
          type: 'scatter3d',
          mode: 'markers',
          x: [[m]],
          y: [[b]],
          z: [[currentLoss]],
          marker: {
            size: 8,
            color: 'red',
            symbol: 'circle',
          },
          name: 'Current (m, b)',
        },
      ];

      const layout = {
        title: 'Loss Landscape',
        scene: {
          xaxis: { title: 'm (slope)' },
          yaxis: { title: 'b (intercept)' },
          zaxis: { title: 'Loss' },
          camera: {
            eye: { x: 1.5, y: 1.5, z: 1.5 },
          },
        },
        margin: { l: 0, r: 0, t: 30, b: 0 },
        height: 400,
      };

      if (lossLandscapeRef.current) {
        Plotly.newPlot(lossLandscapeRef.current, data, layout, { responsive: true });
      }
    });
  }, [showLossLandscape, trainPoints, m, b, lossType]);

  // Data generators
  const generatePerfectLine = () => {
    const points: DataPoint[] = [];
    for (let i = 0; i < 20; i++) {
      const x = -5 + (i / 19) * 10;
      const y = 1.5 * x + 2;
      points.push({ x, y, isTrain: true });
    }
    setDataPoints(points);
  };

  const generateNoisyLine = () => {
    const points: DataPoint[] = [];
    for (let i = 0; i < 30; i++) {
      const x = -5 + (i / 29) * 10;
      const y = 1.5 * x + 2 + (Math.random() - 0.5) * 2;
      points.push({ x, y, isTrain: true });
    }
    setDataPoints(points);
  };

  const generateWithOutlier = () => {
    const points: DataPoint[] = [];
    for (let i = 0; i < 20; i++) {
      const x = -5 + (i / 19) * 10;
      const y = 1.5 * x + 2;
      points.push({ x, y, isTrain: true });
    }
    // Add outlier
    points.push({ x: 3, y: 15, isTrain: true });
    setDataPoints(points);
  };

  const generateNonLinear = () => {
    const points: DataPoint[] = [];
    for (let i = 0; i < 30; i++) {
      const x = -5 + (i / 29) * 10;
      const y = 0.3 * x * x - 2 * x + 1 + (Math.random() - 0.5) * 0.5;
      points.push({ x, y, isTrain: true });
    }
    setDataPoints(points);
  };

  const generateVerticalSpread = () => {
    const points: DataPoint[] = [];
    for (let i = 0; i < 20; i++) {
      const x = -2 + (i / 19) * 4;
      const y = 2 + (Math.random() - 0.5) * 10;
      points.push({ x, y, isTrain: true });
    }
    setDataPoints(points);
  };

  const clearAll = () => {
    setDataPoints([]);
    setM(1.0);
    setB(0.0);
    setStepCount(0);
    setLossHistory([]);
    setGradients(null);
  };

  // Handle auto-fit button
  const handleAutoFit = () => {
    if (trainPoints.length >= 2) {
      const { m: newM, b: newB } = normalEquation(trainPoints);
      setM(newM);
      setB(newB);
      setMode('auto-fit');
      setStepCount(0);
      setLossHistory([]);
    }
  };

  // Mini loss chart component
  const LossChart = ({ data, width: w, height: h }: { data: number[]; width: number; height: number }) => {
    if (data.length === 0) return null;

    const maxLoss = Math.max(...data);
    const minLoss = Math.min(...data);
    const range = maxLoss - minLoss || 1;

    return (
      <svg width={w} height={h} className="border border-gray-200 rounded">
        <polyline
          points={data
            .map((loss, i) => {
              const x = (i / (data.length - 1 || 1)) * (w - 20) + 10;
              const y = h - 10 - ((loss - minLoss) / range) * (h - 20);
              return `${x},${y}`;
            })
            .join(' ')}
          fill="none"
          stroke="#3b82f6"
          strokeWidth="2"
        />
      </svg>
    );
  };

  // Lesson actions
  const handleLessonClick = useCallback((lessonId: string | null) => {
    setActiveLesson(lessonId);
    if (lessonId) {
      switch (lessonId) {
        case 'line':
          setShowGradients(false);
          setShowResiduals(false);
          break;
        case 'prediction':
          setShowResiduals(false);
          break;
        case 'residuals':
          setShowResiduals(true);
          setShowSquaredErrors(false);
          break;
        case 'loss':
          setShowResiduals(true);
          setShowSquaredErrors(true);
          break;
        case 'auto-fit':
          handleAutoFit();
          break;
        case 'gradient-descent':
          setMode('gradient-descent');
          setShowGradients(true);
          break;
        case 'outliers':
          generateWithOutlier();
          setLossType('mse');
          break;
        case 'limitations':
          generateNonLinear();
          break;
      }
    }
  }, [generateWithOutlier, generateNonLinear, handleAutoFit]);

  return (
    <div className="h-full flex flex-col bg-gray-50">
      <div className={`flex-1 flex gap-3 p-3 min-h-0 overflow-hidden ${showLossLandscape ? 'flex-col' : ''}`}>
        <div className={`flex gap-3 flex-1 min-h-0 ${showLossLandscape ? 'flex-row' : ''}`}>
        {/* Lesson Sidebar */}
        <LessonSidebar
          activeLesson={activeLesson}
          onLessonClick={handleLessonClick}
        />

        {/* Left Sidebar - Controls */}
        <div className="w-72 flex-shrink-0 overflow-y-auto bg-white rounded-lg shadow-sm p-4 space-y-4">
          <h2 className="text-lg font-bold text-gray-800 mb-4">Controls</h2>

          {/* Mode Toggle */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">Mode</label>
            <div className="flex gap-1">
              <button
                onClick={() => setMode('manual')}
                className={`flex-1 px-3 py-2 text-sm rounded ${
                  mode === 'manual'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                Manual
              </button>
              <button
                onClick={handleAutoFit}
                className={`flex-1 px-3 py-2 text-sm rounded ${
                  mode === 'auto-fit'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                Auto Fit
              </button>
              <button
                onClick={() => setMode('gradient-descent')}
                className={`flex-1 px-3 py-2 text-sm rounded ${
                  mode === 'gradient-descent'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                GD
              </button>
            </div>
          </div>

          {/* Sliders */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Slope (m): {m.toFixed(3)}
            </label>
            <input
              type="range"
              min="-5"
              max="5"
              step="0.01"
              value={m}
              onChange={(e) => {
                setM(parseFloat(e.target.value));
                setMode('manual');
              }}
              className="w-full"
            />
          </div>

          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Intercept (b): {b.toFixed(3)}
            </label>
            <input
              type="range"
              min="-10"
              max="10"
              step="0.01"
              value={b}
              onChange={(e) => {
                setB(parseFloat(e.target.value));
                setMode('manual');
              }}
              className="w-full"
            />
          </div>

          {/* Gradient Descent Controls */}
          {mode === 'gradient-descent' && (
            <div className="space-y-2 border-t pt-2">
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  Learning Rate: {learningRate.toFixed(3)}
                </label>
                <input
                  type="range"
                  min="0.001"
                  max="1"
                  step="0.001"
                  value={learningRate}
                  onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
              <div className="flex gap-2">
                <button
                  onClick={performGradientDescentStep}
                  className="flex-1 px-3 py-2 bg-blue-600 text-white text-sm rounded hover:bg-blue-700"
                >
                  Step
                </button>
                <button
                  onClick={() => setIsAnimating(!isAnimating)}
                  className={`flex-1 px-3 py-2 text-sm rounded ${
                    isAnimating
                      ? 'bg-red-600 text-white hover:bg-red-700'
                      : 'bg-green-600 text-white hover:bg-green-700'
                  }`}
                >
                  {isAnimating ? 'Pause' : 'Play'}
                </button>
              </div>
              <div className="text-xs text-gray-500">
                Steps: {stepCount}
              </div>
            </div>
          )}

          {/* Loss Type */}
          <div className="border-t pt-2">
            <label className="block text-sm font-semibold text-gray-700 mb-2">Loss Type</label>
            <div className="flex gap-1">
              <button
                onClick={() => setLossType('mse')}
                className={`flex-1 px-3 py-2 text-sm rounded ${
                  lossType === 'mse'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                MSE
              </button>
              <button
                onClick={() => setLossType('mae')}
                className={`flex-1 px-3 py-2 text-sm rounded ${
                  lossType === 'mae'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                MAE
              </button>
              <button
                onClick={() => setLossType('huber')}
                className={`flex-1 px-3 py-2 text-sm rounded ${
                  lossType === 'huber'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                Huber
              </button>
            </div>
          </div>

          {/* Data Generators */}
          <div className="border-t pt-2">
            <label className="block text-sm font-semibold text-gray-700 mb-2">Data Generators</label>
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={generatePerfectLine}
                className="px-2 py-1.5 text-xs bg-green-100 text-green-700 rounded hover:bg-green-200"
              >
                Perfect
              </button>
              <button
                onClick={generateNoisyLine}
                className="px-2 py-1.5 text-xs bg-blue-100 text-blue-700 rounded hover:bg-blue-200"
              >
                Noisy
              </button>
              <button
                onClick={generateWithOutlier}
                className="px-2 py-1.5 text-xs bg-orange-100 text-orange-700 rounded hover:bg-orange-200"
              >
                Outlier
              </button>
              <button
                onClick={generateNonLinear}
                className="px-2 py-1.5 text-xs bg-purple-100 text-purple-700 rounded hover:bg-purple-200"
              >
                Non-linear
              </button>
              <button
                onClick={generateVerticalSpread}
                className="px-2 py-1.5 text-xs bg-pink-100 text-pink-700 rounded hover:bg-pink-200"
              >
                Spread
              </button>
              <button
                onClick={clearAll}
                className="px-2 py-1.5 text-xs bg-red-100 text-red-700 rounded hover:bg-red-200"
              >
                Clear
              </button>
            </div>
          </div>

          {/* Train-Test Split */}
          <div className="border-t pt-2">
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Train-Test Split: {(trainTestSplit * 100).toFixed(0)}%
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={trainTestSplit}
              onChange={(e) => setTrainTestSplit(parseFloat(e.target.value))}
              className="w-full"
            />
            <div className="flex gap-2 mt-1 text-xs">
              <span className="text-blue-600">Train: {trainPoints.length}</span>
              <span className="text-orange-600">Test: {testPoints.length}</span>
            </div>
          </div>

          {/* Visualization Options */}
          <div className="border-t pt-2">
            <label className="block text-sm font-semibold text-gray-700 mb-2">Visualization</label>
            <div className="space-y-1">
              <label className="flex items-center gap-2 text-sm text-gray-700">
                <input
                  type="checkbox"
                  checked={showResiduals}
                  onChange={(e) => setShowResiduals(e.target.checked)}
                  className="rounded"
                />
                Show Residuals
              </label>
              <label className="flex items-center gap-2 text-sm text-gray-700">
                <input
                  type="checkbox"
                  checked={showSquaredErrors}
                  onChange={(e) => setShowSquaredErrors(e.target.checked)}
                  className="rounded"
                />
                Show Squared Errors
              </label>
              <label className="flex items-center gap-2 text-sm text-gray-700">
                <input
                  type="checkbox"
                  checked={showGradients}
                  onChange={(e) => setShowGradients(e.target.checked)}
                  className="rounded"
                />
                Show Gradients
              </label>
              <label className="flex items-center gap-2 text-sm text-gray-700">
                <input
                  type="checkbox"
                  checked={showLossLandscape}
                  onChange={(e) => setShowLossLandscape(e.target.checked)}
                  className="rounded"
                />
                Loss Landscape
              </label>
            </div>
          </div>
        </div>

        {/* Center - Main Graph */}
        <div className="flex-1 flex flex-col items-center justify-center min-w-0 min-h-0">
          <div className="text-sm text-gray-500 mb-2">
            Click to add • Drag to move • Right-click to delete • ({dataPoints.length} points)
          </div>
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
              maxWidth: 'min(calc(100vh - 200px), 100%)',
              maxHeight: 'calc(100vh - 200px)',
            }}
          />
        </div>

        {/* Right Sidebar - Metrics */}
        <div className="w-64 flex-shrink-0 overflow-y-auto bg-white rounded-lg shadow-sm p-4 space-y-4">
          <h2 className="text-lg font-bold text-gray-800 mb-4">Metrics</h2>

          <div>
            <div className="text-sm font-semibold text-gray-700 mb-1">MSE</div>
            <div className="text-2xl font-mono text-blue-600">{metrics.mse.toFixed(4)}</div>
          </div>

          <div>
            <div className="text-sm font-semibold text-gray-700 mb-1">MAE</div>
            <div className="text-2xl font-mono text-green-600">{metrics.mae.toFixed(4)}</div>
          </div>

          <div>
            <div className="text-sm font-semibold text-gray-700 mb-1">R²</div>
            <div className="text-2xl font-mono text-purple-600">{metrics.r2.toFixed(4)}</div>
          </div>

          {lossHistory.length > 0 && (
            <div className="border-t pt-2">
              <div className="text-sm font-semibold text-gray-700 mb-2">Loss History</div>
              <LossChart data={lossHistory} width={200} height={100} />
            </div>
          )}

          {testPoints.length > 0 && (
            <div className="border-t pt-2">
              <div className="text-sm font-semibold text-gray-700 mb-2">Test Metrics</div>
              <div className="text-xs text-gray-600 space-y-1">
                <div>MSE: {computeMSE(testPoints, m, b).toFixed(4)}</div>
                <div>MAE: {computeMAE(testPoints, m, b).toFixed(4)}</div>
                <div>R²: {computeR2(testPoints, m, b).toFixed(4)}</div>
              </div>
            </div>
          )}
        </div>
        </div>

        {/* Loss Landscape Panel */}
        {showLossLandscape && trainPoints.length >= 2 && (
          <div className="flex-shrink-0 bg-white rounded-lg shadow-sm p-4" style={{ height: '400px' }}>
            <h3 className="text-lg font-bold text-gray-800 mb-2">Loss Landscape</h3>
            <div ref={lossLandscapeRef} style={{ width: '100%', height: '100%' }} />
          </div>
        )}
      </div>
    </div>
  );
}


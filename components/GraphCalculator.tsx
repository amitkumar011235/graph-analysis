'use client';

import React, { useRef, useEffect, useState, useCallback, useMemo } from 'react';

interface Expression {
  id: string;
  label: string; // y1, y2, y3, etc.
  text: string;
  color: string;
  visible: boolean;
}

interface Parameter {
  name: string;
  value: number;
  min: number;
  max: number;
}

const COLORS = [
  '#3b82f6', // blue
  '#ef4444', // red
  '#10b981', // green
  '#f59e0b', // amber
  '#8b5cf6', // purple
  '#ec4899', // pink
  '#06b6d4', // cyan
  '#f97316', // orange
];

// Reserved words that should not be treated as parameters
const RESERVED_WORDS = new Set([
  'x', 'sin', 'cos', 'tan', 'log', 'ln', 'sqrt', 'abs', 'exp', 
  'floor', 'ceil', 'round', 'pi', 'asin', 'acos', 'atan', 'e',
  'relu', 'sigmoid', 'softmax', 'tanh', 'leakyrelu', 'swish', 'gelu'
]);

// Custom activation functions
const relu = (x: number) => Math.max(0, x);
const leakyRelu = (x: number) => x > 0 ? x : 0.01 * x;
const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
const softmax = (x: number) => Math.exp(x) / (1 + Math.exp(x)); // Simplified for single value
const swish = (x: number) => x * sigmoid(x);
const gelu = (x: number) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));

// Extract expression references (y1, y2, etc.) from an expression
function extractExpressionRefs(expr: string): string[] {
  const matches = expr.toLowerCase().match(/y\d+/g) || [];
  return [...new Set(matches)];
}

// Extract parameter names from expression
// Supports: single letters (a, b, w) AND letter+number patterns (w1, b1, a2)
function extractParameters(expr: string, expressionLabels: string[] = []): string[] {
  // Remove function names first
  let cleaned = expr
    .replace(/sin|cos|tan|log|ln|sqrt|abs|exp|floor|ceil|round|asin|acos|atan|pi|relu|sigmoid|softmax|tanh|leakyrelu|swish|gelu/gi, ' ')
    .replace(/y\d+/gi, ' '); // Remove expression references y1, y2, etc.
  
  // Find all parameter patterns:
  // 1. Single letters (not x, e, or y): a, b, w, m, etc.
  // 2. Letter followed by numbers (not y): w1, b1, a2, m3, etc.
  const singleLetters = cleaned.match(/\b[a-df-wz]\b/gi) || []; // Single letters except x, e, y
  const letterNumbers = cleaned.match(/\b[a-wz]\d+\b/gi) || []; // Letter+number except y1, y2, x1
  
  // Combine and deduplicate
  const allMatches = [...singleLetters, ...letterNumbers];
  const params = [...new Set(allMatches.map(p => p.toLowerCase()))]
    .filter(p => !RESERVED_WORDS.has(p));
  
  // Sort: single letters first, then by name
  return params.sort((a, b) => {
    const aHasNum = /\d/.test(a);
    const bHasNum = /\d/.test(b);
    if (aHasNum && !bHasNum) return 1;
    if (!aHasNum && bHasNum) return -1;
    return a.localeCompare(b);
  });
}

// Topologically sort expressions based on dependencies
function topologicalSort(expressions: Expression[]): Expression[] {
  const labelToExpr = new Map<string, Expression>();
  expressions.forEach(e => labelToExpr.set(e.label.toLowerCase(), e));
  
  const visited = new Set<string>();
  const result: Expression[] = [];
  
  function visit(label: string, visiting: Set<string>) {
    if (visited.has(label)) return;
    if (visiting.has(label)) return; // Circular dependency - skip
    
    const expr = labelToExpr.get(label);
    if (!expr) return;
    
    visiting.add(label);
    
    // Visit dependencies first
    const refs = extractExpressionRefs(expr.text);
    for (const ref of refs) {
      visit(ref, visiting);
    }
    
    visiting.delete(label);
    visited.add(label);
    result.push(expr);
  }
  
  for (const expr of expressions) {
    visit(expr.label.toLowerCase(), new Set());
  }
  
  return result;
}

// Safe math evaluation with parameters and expression references
function evaluateExpression(
  expr: string, 
  x: number, 
  params: Record<string, number>,
  exprValues: Record<string, number> = {}
): number | null {
  try {
    let sanitized = expr.trim().toLowerCase();
    
    // Replace expression references (y1, y2, etc.) with their computed values
    // Sort by length (longest first) to handle y10 before y1
    const exprNames = Object.keys(exprValues).sort((a, b) => b.length - a.length);
    for (const name of exprNames) {
      const regex = new RegExp(`\\b${name}\\b`, 'g');
      const value = exprValues[name];
      if (value !== null && isFinite(value)) {
        sanitized = sanitized.replace(regex, `(${value})`);
      } else {
        // If referenced expression has no value, return null
        if (sanitized.match(regex)) {
          return null;
        }
      }
    }
    
    // Get parameter names sorted by length (longest first) to avoid partial replacements
    const paramNames = Object.keys(params).sort((a, b) => b.length - a.length);
    
    // Replace parameters with their values
    for (const name of paramNames) {
      const regex = new RegExp(`\\b${name}\\b`, 'g');
      sanitized = sanitized.replace(regex, `(${params[name]})`);
    }
    
    // Handle implicit multiplication: 2x → 2*x, 3sin(x) → 3*sin(x), (x)(x) → (x)*(x), x(2) → x*(2)
    sanitized = sanitized
      // Number followed by x: 2x → 2*x
      .replace(/(\d)([x])/gi, '$1*$2')
      // Number followed by opening paren: 2( → 2*(
      .replace(/(\d)\(/g, '$1*(')
      // x followed by opening paren: x( → x*(
      .replace(/([x])\(/gi, '$1*(')
      // Closing paren followed by opening paren: )( → )*(
      .replace(/\)\(/g, ')*(')
      // Closing paren followed by x: )x → )*x
      .replace(/\)([x])/gi, ')*$1')
      // Closing paren followed by number: )2 → )*2
      .replace(/\)(\d)/g, ')*$1')
      // Number followed by function name: 2sin → 2*sin
      .replace(/(\d)(sin|cos|tan|log|ln|sqrt|abs|exp|floor|ceil|round|asin|acos|atan|relu|sigmoid|softmax|tanh|leakyrelu|swish|gelu)/gi, '$1*$2')
      // x followed by number (rare but possible): x2 → x*2
      .replace(/([x])(\d)/gi, '$1*$2');
    
    // Replace activation functions (must be done before standard math functions)
    // Order matters: leakyrelu before relu to avoid partial replacement
    sanitized = sanitized
      .replace(/leakyrelu/gi, '((x)=>x>0?x:0.01*x)')
      .replace(/relu/gi, '((x)=>Math.max(0,x))')
      .replace(/sigmoid/gi, '((x)=>1/(1+Math.exp(-x)))')
      .replace(/softmax/gi, '((x)=>Math.exp(x)/(1+Math.exp(x)))')
      .replace(/swish/gi, '((x)=>x/(1+Math.exp(-x)))')
      .replace(/gelu/gi, '((x)=>0.5*x*(1+Math.tanh(Math.sqrt(2/Math.PI)*(x+0.044715*Math.pow(x,3)))))')
      .replace(/tanh/gi, 'Math.tanh');
    
    // Replace common math functions and constants
    sanitized = sanitized
      .replace(/\^/g, '**')
      .replace(/sin/g, 'Math.sin')
      .replace(/cos/g, 'Math.cos')
      .replace(/tan/g, 'Math.tan')
      .replace(/log/g, 'Math.log10')
      .replace(/ln/g, 'Math.log')
      .replace(/sqrt/g, 'Math.sqrt')
      .replace(/abs/g, 'Math.abs')
      .replace(/exp/g, 'Math.exp')
      .replace(/floor/g, 'Math.floor')
      .replace(/ceil/g, 'Math.ceil')
      .replace(/round/g, 'Math.round')
      .replace(/pi/gi, 'Math.PI')
      .replace(/e(?![xp])/g, 'Math.E')
      .replace(/asin/g, 'Math.asin')
      .replace(/acos/g, 'Math.acos')
      .replace(/atan/g, 'Math.atan');
    
    // Fix double Math. references (caused by activation functions already having Math.exp, etc.)
    sanitized = sanitized.replace(/Math\.Math\./g, 'Math.');
    
    // Create a function that evaluates the expression
    const fn = new Function('x', `return ${sanitized}`);
    const result = fn(x);
    
    if (typeof result === 'number' && isFinite(result)) {
      return result;
    }
    return null;
  } catch {
    return null;
  }
}

// Evaluate all expressions for a given x value, respecting dependencies
function evaluateAllExpressions(
  expressions: Expression[],
  x: number,
  params: Record<string, number>
): Record<string, number | null> {
  const sortedExprs = topologicalSort(expressions);
  const values: Record<string, number | null> = {};
  
  for (const expr of sortedExprs) {
    if (expr.text.trim()) {
      values[expr.label.toLowerCase()] = evaluateExpression(expr.text, x, params, values as Record<string, number>);
    } else {
      values[expr.label.toLowerCase()] = null;
    }
  }
  
  return values;
}

export default function GraphCalculator() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [expressions, setExpressions] = useState<Expression[]>([
    { id: '1', label: 'y1', text: 'x^2', color: COLORS[0], visible: true },
  ]);
  const [labelCounter, setLabelCounter] = useState(2); // Next label number
  const [parameters, setParameters] = useState<Parameter[]>([]);
  const [xMin, setXMin] = useState(-10);
  const [xMax, setXMax] = useState(10);
  const [yMin, setYMin] = useState(-10);
  const [yMax, setYMax] = useState(10);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [viewStart, setViewStart] = useState({ xMin: -10, xMax: 10, yMin: -10, yMax: 10 });

  const width = 700;
  const height = 700;
  const padding = 50;

  // Get all expression labels for parameter extraction
  const expressionLabels = useMemo(() => 
    expressions.map(e => e.label.toLowerCase()), 
  [expressions]);

  // Extract all parameters from ALL expressions (for storage)
  const allDetectedParams = useMemo(() => {
    const allParams = new Set<string>();
    expressions.forEach(expr => {
      if (expr.text.trim()) {
        extractParameters(expr.text, expressionLabels).forEach(p => allParams.add(p));
      }
    });
    return [...allParams].sort();
  }, [expressions, expressionLabels]);

  // Extract parameters from VISIBLE expressions only (for display)
  const visibleParams = useMemo(() => {
    const params = new Set<string>();
    expressions.forEach(expr => {
      if (expr.visible && expr.text.trim()) {
        extractParameters(expr.text, expressionLabels).forEach(p => params.add(p));
      }
    });
    return [...params].sort();
  }, [expressions, expressionLabels]);

  // Auto-add new parameters when detected (keep all params in storage)
  useEffect(() => {
    const existingNames = new Set(parameters.map(p => p.name));
    const newParams = allDetectedParams.filter(name => !existingNames.has(name));
    
    if (newParams.length > 0) {
      setParameters(prev => [
        ...prev,
        ...newParams.map(name => ({
          name,
          value: 1,
          min: -10,
          max: 10,
        }))
      ]);
    }
    
    // Remove parameters that are no longer used in ANY expression
    const usedParams = new Set(allDetectedParams);
    setParameters(prev => prev.filter(p => usedParams.has(p.name)));
  }, [allDetectedParams]);

  // Get only the parameters that should be displayed (from visible expressions)
  const displayedParameters = useMemo(() => {
    const visibleSet = new Set(visibleParams);
    return parameters.filter(p => visibleSet.has(p.name));
  }, [parameters, visibleParams]);

  // Convert parameters array to object for evaluation
  const paramValues = useMemo(() => {
    const values: Record<string, number> = {};
    parameters.forEach(p => {
      values[p.name] = p.value;
    });
    return values;
  }, [parameters]);

  // Convert data coordinates to canvas coordinates
  const dataToCanvas = useCallback((x: number, y: number) => {
    const canvasX = ((x - xMin) / (xMax - xMin)) * (width - 2 * padding) + padding;
    const canvasY = height - padding - ((y - yMin) / (yMax - yMin)) * (height - 2 * padding);
    return { x: canvasX, y: canvasY };
  }, [xMin, xMax, yMin, yMax]);

  // Convert canvas coordinates to data coordinates
  const canvasToData = useCallback((canvasX: number, canvasY: number) => {
    const x = ((canvasX - padding) / (width - 2 * padding)) * (xMax - xMin) + xMin;
    const y = yMax - ((canvasY - padding) / (height - 2 * padding)) * (yMax - yMin);
    return { x, y };
  }, [xMin, xMax, yMin, yMax]);

  // Draw the graph
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);

    // Draw grid
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;

    // Calculate grid spacing
    const xRange = xMax - xMin;
    const yRange = yMax - yMin;
    const xGridStep = Math.pow(10, Math.floor(Math.log10(xRange / 5)));
    const yGridStep = Math.pow(10, Math.floor(Math.log10(yRange / 5)));

    // Vertical grid lines
    const xStart = Math.ceil(xMin / xGridStep) * xGridStep;
    for (let x = xStart; x <= xMax; x += xGridStep) {
      const canvasX = dataToCanvas(x, 0).x;
      ctx.beginPath();
      ctx.moveTo(canvasX, padding);
      ctx.lineTo(canvasX, height - padding);
      ctx.stroke();
    }

    // Horizontal grid lines
    const yStart = Math.ceil(yMin / yGridStep) * yGridStep;
    for (let y = yStart; y <= yMax; y += yGridStep) {
      const canvasY = dataToCanvas(0, y).y;
      ctx.beginPath();
      ctx.moveTo(padding, canvasY);
      ctx.lineTo(width - padding, canvasY);
      ctx.stroke();
    }

    // Draw axes
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 2;

    // X axis (if visible)
    if (yMin <= 0 && yMax >= 0) {
      const axisY = dataToCanvas(0, 0).y;
      ctx.beginPath();
      ctx.moveTo(padding, axisY);
      ctx.lineTo(width - padding, axisY);
      ctx.stroke();
    }

    // Y axis (if visible)
    if (xMin <= 0 && xMax >= 0) {
      const axisX = dataToCanvas(0, 0).x;
      ctx.beginPath();
      ctx.moveTo(axisX, padding);
      ctx.lineTo(axisX, height - padding);
      ctx.stroke();
    }

    // Draw axis labels
    ctx.fillStyle = '#6b7280';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    // X axis labels
    for (let x = xStart; x <= xMax; x += xGridStep) {
      if (Math.abs(x) > xGridStep / 10) {
        const canvasX = dataToCanvas(x, 0).x;
        const labelY = yMin <= 0 && yMax >= 0 ? dataToCanvas(0, 0).y + 5 : height - padding + 5;
        ctx.fillText(x.toFixed(Math.max(0, -Math.floor(Math.log10(xGridStep)))), canvasX, labelY);
      }
    }

    // Y axis labels
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let y = yStart; y <= yMax; y += yGridStep) {
      if (Math.abs(y) > yGridStep / 10) {
        const canvasY = dataToCanvas(0, y).y;
        const labelX = xMin <= 0 && xMax >= 0 ? dataToCanvas(0, 0).x - 5 : padding - 5;
        ctx.fillText(y.toFixed(Math.max(0, -Math.floor(Math.log10(yGridStep)))), labelX, canvasY);
      }
    }

    // Draw border
    ctx.strokeStyle = '#d1d5db';
    ctx.lineWidth = 1;
    ctx.strokeRect(padding, padding, width - 2 * padding, height - 2 * padding);

    // Draw expressions
    const resolution = (width - 2 * padding) * 2; // 2 points per pixel for smoothness

    // Pre-compute all expression values for each x to handle dependencies
    const expressionPaths: Map<string, { x: number; y: number | null }[]> = new Map();
    
    for (let i = 0; i <= resolution; i++) {
      const x = xMin + (i / resolution) * (xMax - xMin);
      const allValues = evaluateAllExpressions(expressions, x, paramValues);
      
      expressions.forEach((expr) => {
        if (!expressionPaths.has(expr.label)) {
          expressionPaths.set(expr.label, []);
        }
        expressionPaths.get(expr.label)!.push({
          x,
          y: allValues[expr.label.toLowerCase()]
        });
      });
    }

    // Draw each visible expression
    expressions.forEach((expr) => {
      if (!expr.visible || !expr.text.trim()) return;

      const points = expressionPaths.get(expr.label) || [];
      
      ctx.strokeStyle = expr.color;
      ctx.lineWidth = 2.5;
      ctx.beginPath();

      let started = false;
      let prevY: number | null = null;

      for (const point of points) {
        const y = point.y;

        if (y !== null && y >= yMin - (yMax - yMin) && y <= yMax + (yMax - yMin)) {
          const canvas = dataToCanvas(point.x, y);
          
          // Check for discontinuity (large jump)
          if (prevY !== null && Math.abs(y - prevY) > (yMax - yMin) * 0.5) {
            started = false;
          }

          if (!started) {
            ctx.moveTo(canvas.x, canvas.y);
            started = true;
          } else {
            ctx.lineTo(canvas.x, canvas.y);
          }
          prevY = y;
        } else {
          started = false;
          prevY = null;
        }
      }

      ctx.stroke();
    });

    // Draw origin label
    if (xMin <= 0 && xMax >= 0 && yMin <= 0 && yMax >= 0) {
      const origin = dataToCanvas(0, 0);
      ctx.fillStyle = '#374151';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'top';
      ctx.fillText('0', origin.x - 5, origin.y + 5);
    }
  }, [expressions, xMin, xMax, yMin, yMax, dataToCanvas, paramValues]);

  useEffect(() => {
    draw();
  }, [draw]);

  // Add new expression
  const addExpression = () => {
    const newId = Date.now().toString();
    const colorIndex = expressions.length % COLORS.length;
    const newLabel = `y${labelCounter}`;
    setExpressions([
      ...expressions,
      { id: newId, label: newLabel, text: '', color: COLORS[colorIndex], visible: true },
    ]);
    setLabelCounter(prev => prev + 1);
  };

  // Update expression text
  const updateExpression = (id: string, text: string) => {
    setExpressions(expressions.map((e) => (e.id === id ? { ...e, text } : e)));
  };

  // Update expression color
  const updateColor = (id: string, color: string) => {
    setExpressions(expressions.map((e) => (e.id === id ? { ...e, color } : e)));
  };

  // Toggle expression visibility
  const toggleVisibility = (id: string) => {
    setExpressions(expressions.map((e) => (e.id === id ? { ...e, visible: !e.visible } : e)));
  };

  // Delete expression
  const deleteExpression = (id: string) => {
    if (expressions.length > 1) {
      setExpressions(expressions.filter((e) => e.id !== id));
    }
  };

  // Zoom in/out
  const zoom = (factor: number) => {
    const centerX = (xMin + xMax) / 2;
    const centerY = (yMin + yMax) / 2;
    const newRangeX = (xMax - xMin) * factor;
    const newRangeY = (yMax - yMin) * factor;
    setXMin(centerX - newRangeX / 2);
    setXMax(centerX + newRangeX / 2);
    setYMin(centerY - newRangeY / 2);
    setYMax(centerY + newRangeY / 2);
  };

  // Reset view
  const resetView = () => {
    setXMin(-10);
    setXMax(10);
    setYMin(-10);
    setYMax(10);
  };

  // Handle mouse wheel for zooming
  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const factor = e.deltaY > 0 ? 1.1 : 0.9;
    zoom(factor);
  };

  // Handle mouse drag for panning
  const handleMouseDown = (e: React.MouseEvent) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    
    setIsDragging(true);
    setDragStart({ x: e.clientX, y: e.clientY });
    setViewStart({ xMin, xMax, yMin, yMax });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;

    const dx = e.clientX - dragStart.x;
    const dy = e.clientY - dragStart.y;
    
    const scaleX = (viewStart.xMax - viewStart.xMin) / (width - 2 * padding);
    const scaleY = (viewStart.yMax - viewStart.yMin) / (height - 2 * padding);
    
    setXMin(viewStart.xMin - dx * scaleX);
    setXMax(viewStart.xMax - dx * scaleX);
    setYMin(viewStart.yMin + dy * scaleY);
    setYMax(viewStart.yMax + dy * scaleY);
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  // Get canvas coordinates with scaling
  const getCanvasCoords = (e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    
    const rect = canvas.getBoundingClientRect();
    const scaleX = width / rect.width;
    const scaleY = height / rect.height;
    
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  };

  return (
    <div className="flex gap-3 h-full">
      {/* Left Sidebar - Expressions */}
      <div className="w-72 flex-shrink-0 bg-white border border-gray-200 rounded-lg p-3 flex flex-col h-full overflow-hidden">
        <h2 className="text-lg font-bold text-gray-800 mb-3 flex-shrink-0">Expressions</h2>
        
        {/* Scrollable content area */}
        <div className="flex-1 overflow-y-auto min-h-0">
          {/* Expressions List */}
          <div className="space-y-2">
            {expressions.map((expr) => (
              <div
                key={expr.id}
                className="flex items-center gap-2 p-2 bg-gray-50 rounded-md border border-gray-200"
              >
                {/* Visibility toggle */}
                <button
                  onClick={() => toggleVisibility(expr.id)}
                  className="flex-shrink-0 text-gray-400 hover:text-gray-600 transition-colors"
                  title={expr.visible ? 'Hide graph' : 'Show graph'}
                >
                  {expr.visible ? (
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                    </svg>
                  ) : (
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
                    </svg>
                  )}
                </button>
                
                {/* Color picker */}
                <label className="relative flex-shrink-0 cursor-pointer" title="Change color">
                  <div 
                    className="w-5 h-5 rounded border-2 border-gray-300 hover:border-gray-400 transition-colors"
                    style={{ backgroundColor: expr.color }}
                  />
                  <input
                    type="color"
                    value={expr.color}
                    onChange={(e) => updateColor(expr.id, e.target.value)}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  />
                </label>
                
                <div className="flex-1 flex items-center gap-1 min-w-0">
                  <span 
                    className="text-sm flex-shrink-0 font-medium"
                    style={{ color: expr.color }}
                  >
                    {expr.label} =
                  </span>
                  <input
                    type="text"
                    value={expr.text}
                    onChange={(e) => updateExpression(expr.id, e.target.value)}
                    placeholder={`e.g., x^2, sin(x)${expressions.length > 1 ? ', y1+x' : ''}`}
                    className="flex-1 min-w-0 px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500 text-gray-900 bg-white placeholder-gray-400"
                  />
                </div>
                {expressions.length > 1 && (
                  <button
                    onClick={() => deleteExpression(expr.id)}
                    className="text-gray-400 hover:text-red-500 text-lg flex-shrink-0"
                    title="Delete"
                  >
                    ×
                  </button>
                )}
              </div>
            ))}
          </div>

          <button
            onClick={addExpression}
            className="mt-3 w-full px-3 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-colors text-sm"
          >
            + Add Expression
          </button>

          {/* Parameter Sliders */}
          {displayedParameters.length > 0 && (
            <div className="mt-4 pt-4 border-t border-gray-200">
              <h3 className="text-sm font-semibold text-gray-700 mb-2">
                Parameters
                <span className="text-xs font-normal text-gray-400 ml-2">
                  (auto-detected)
                </span>
              </h3>
              <div className="space-y-3">
                {displayedParameters.map((param) => (
                  <div key={param.name} className="bg-gray-50 p-2 rounded-md border border-gray-200">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-sm font-medium text-blue-600">{param.name}</span>
                      <input
                        type="number"
                        value={param.value}
                        onChange={(e) => {
                          const newVal = parseFloat(e.target.value);
                          if (!isNaN(newVal)) {
                            setParameters(prev => prev.map(p => 
                              p.name === param.name ? { ...p, value: newVal } : p
                            ));
                          }
                        }}
                        className="w-16 px-1 py-0.5 text-xs text-right border border-gray-300 rounded text-gray-900 bg-white"
                        step="0.1"
                      />
                    </div>
                    <input
                      type="range"
                      min={param.min}
                      max={param.max}
                      step="0.1"
                      value={param.value}
                      onChange={(e) => {
                        setParameters(prev => prev.map(p => 
                          p.name === param.name ? { ...p, value: parseFloat(e.target.value) } : p
                        ));
                      }}
                      className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-500"
                    />
                    <div className="flex justify-between text-xs text-gray-400 mt-1">
                      <button
                        onClick={() => setParameters(prev => prev.map(p => 
                          p.name === param.name ? { ...p, min: p.min - 10 } : p
                        ))}
                        className="hover:text-gray-600"
                      >
                        {param.min}
                      </button>
                      <button
                        onClick={() => setParameters(prev => prev.map(p => 
                          p.name === param.name ? { ...p, max: p.max + 10 } : p
                        ))}
                        className="hover:text-gray-600"
                      >
                        {param.max}
                      </button>
                    </div>
                  </div>
                ))}
              </div>
              <p className="text-xs text-gray-400 mt-2">
                Parameters: a, b, w or w1, b1, a2
              </p>
            </div>
          )}

          {/* View Controls */}
          <div className="mt-4 pt-4 border-t border-gray-200">
            <h3 className="text-sm font-semibold text-gray-700 mb-2">View Controls</h3>
            <div className="grid grid-cols-3 gap-2">
              <button
                onClick={() => zoom(0.8)}
                className="px-2 py-1 bg-gray-100 text-gray-700 rounded hover:bg-gray-200 text-sm"
              >
                Zoom +
              </button>
              <button
                onClick={resetView}
                className="px-2 py-1 bg-gray-100 text-gray-700 rounded hover:bg-gray-200 text-sm"
              >
                Reset
              </button>
              <button
                onClick={() => zoom(1.25)}
                className="px-2 py-1 bg-gray-100 text-gray-700 rounded hover:bg-gray-200 text-sm"
              >
                Zoom −
              </button>
            </div>
            <p className="text-xs text-gray-400 mt-2">
              Scroll to zoom • Drag to pan
            </p>
          </div>

          {/* Function Reference */}
          <div className="mt-4 pt-4 border-t border-gray-200">
            <h3 className="text-sm font-semibold text-gray-700 mb-2">Math Functions</h3>
            <div className="text-xs text-gray-500 space-y-1">
              <p><code className="bg-gray-100 px-1 rounded">^</code> power (x^2)</p>
              <p><code className="bg-gray-100 px-1 rounded">sin, cos, tan</code></p>
              <p><code className="bg-gray-100 px-1 rounded">sqrt, abs, log, ln</code></p>
              <p><code className="bg-gray-100 px-1 rounded">pi, e</code> constants</p>
            </div>
            <h3 className="text-sm font-semibold text-gray-700 mt-3 mb-2">Activations</h3>
            <div className="text-xs text-gray-500 space-y-1">
              <p><code className="bg-gray-100 px-1 rounded">relu, leakyrelu</code></p>
              <p><code className="bg-gray-100 px-1 rounded">sigmoid, tanh</code></p>
              <p><code className="bg-gray-100 px-1 rounded">softmax, swish, gelu</code></p>
            </div>
            <h3 className="text-sm font-semibold text-gray-700 mt-3 mb-2">Compose</h3>
            <div className="text-xs text-gray-500 space-y-1">
              <p>Use <code className="bg-gray-100 px-1 rounded">y1, y2...</code> to reference other expressions</p>
              <p className="text-gray-400 italic">e.g., y2 = sigmoid(y1)</p>
            </div>
          </div>
        </div>
      </div>

      {/* Canvas Area */}
      <div className="flex-1 flex flex-col items-center justify-center min-w-0 min-h-0">
        <canvas
          ref={canvasRef}
          width={width}
          height={height}
          onWheel={handleWheel}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          className="border border-gray-300 rounded-lg bg-white shadow-md cursor-grab active:cursor-grabbing"
          style={{
            width: '100%',
            height: '100%',
            maxWidth: 'min(calc(100vh - 120px), 100%)',
            maxHeight: 'calc(100vh - 120px)',
            aspectRatio: '1',
          }}
        />
        <p className="text-xs text-gray-400 mt-2">
          x: [{xMin.toFixed(1)}, {xMax.toFixed(1)}] • y: [{yMin.toFixed(1)}, {yMax.toFixed(1)}]
        </p>
      </div>
    </div>
  );
}


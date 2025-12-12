// Utility functions for graph analysis and calculus operations

export interface Expression {
  id: string;
  label: string;
  text: string;
  color: string;
  visible: boolean;
}

// Evaluate expression at a point (reuse from GraphCalculator)
function evaluateExpression(
  expr: string,
  x: number,
  params: Record<string, number>,
  exprValues: Record<string, number> = {}
): number | null {
  try {
    let sanitized = expr.trim().toLowerCase();
    
    // Replace expression references (y1, y2, etc.) with their computed values
    const exprNames = Object.keys(exprValues).sort((a, b) => b.length - a.length);
    for (const name of exprNames) {
      const regex = new RegExp(`\\b${name}\\b`, 'g');
      const value = exprValues[name];
      if (value !== null && isFinite(value)) {
        sanitized = sanitized.replace(regex, `(${value})`);
      } else {
        if (sanitized.match(regex)) {
          return null;
        }
      }
    }
    
    // Replace parameters
    const paramNames = Object.keys(params).sort((a, b) => b.length - a.length);
    for (const name of paramNames) {
      const regex = new RegExp(`\\b${name}\\b`, 'g');
      sanitized = sanitized.replace(regex, `(${params[name]})`);
    }
    
    // Handle implicit multiplication
    sanitized = sanitized
      .replace(/(\d)([x])/gi, '$1*$2')
      .replace(/(\d)\(/g, '$1*(')
      .replace(/([x])\(/gi, '$1*(')
      .replace(/\)\(/g, ')*(')
      .replace(/\)([x])/gi, ')*$1')
      .replace(/\)(\d)/g, ')*$1')
      .replace(/(\d)(sin|cos|tan|log|ln|sqrt|abs|exp|floor|ceil|round|asin|acos|atan|relu|sigmoid|softmax|tanh|leakyrelu|swish|gelu)/gi, '$1*$2')
      .replace(/([x])(\d)/gi, '$1*$2');
    
    // Replace activation functions
    sanitized = sanitized
      .replace(/leakyrelu/gi, '((x)=>x>0?x:0.01*x)')
      .replace(/relu/gi, '((x)=>Math.max(0,x))')
      .replace(/sigmoid/gi, '((x)=>1/(1+Math.exp(-x)))')
      .replace(/softmax/gi, '((x)=>Math.exp(x)/(1+Math.exp(x)))')
      .replace(/swish/gi, '((x)=>x/(1+Math.exp(-x)))')
      .replace(/gelu/gi, '((x)=>0.5*x*(1+Math.tanh(Math.sqrt(2/Math.PI)*(x+0.044715*Math.pow(x,3)))))')
      .replace(/tanh/gi, 'Math.tanh');
    
    // Replace math functions
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
    
    // Fix double Math references
    sanitized = sanitized.replace(/Math\.Math\./g, 'Math.');
    
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

// Evaluate all expressions (for dependency resolution)
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

// Topological sort for expression dependencies
function topologicalSort(expressions: Expression[]): Expression[] {
  const labelToExpr = new Map<string, Expression>();
  expressions.forEach(e => labelToExpr.set(e.label.toLowerCase(), e));
  
  const visited = new Set<string>();
  const result: Expression[] = [];
  
  function visit(label: string, visiting: Set<string>) {
    if (visited.has(label)) return;
    if (visiting.has(label)) return;
    
    const expr = labelToExpr.get(label);
    if (!expr) return;
    
    visiting.add(label);
    
    const refs = expr.text.toLowerCase().match(/y\d+/g) || [];
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

// Numerical derivative using finite differences
export function calculateDerivative(
  expr: string,
  x: number,
  params: Record<string, number>,
  exprValues: Record<string, number> = {},
  h: number = 1e-5
): number | null {
  const f1 = evaluateExpression(expr, x + h, params, exprValues);
  const f2 = evaluateExpression(expr, x - h, params, exprValues);
  
  if (f1 === null || f2 === null) return null;
  
  return (f1 - f2) / (2 * h);
}

// Numerical integral using Simpson's rule
export function calculateIntegral(
  expr: string,
  xMin: number,
  xMax: number,
  params: Record<string, number>,
  exprValues: Record<string, number> = {},
  n: number = 1000
): number | null {
  if (xMin >= xMax) return null;
  
  const h = (xMax - xMin) / n;
  let sum = 0;
  
  for (let i = 0; i <= n; i++) {
    const x = xMin + i * h;
    const y = evaluateExpression(expr, x, params, exprValues);
    
    if (y === null) return null;
    
    if (i === 0 || i === n) {
      sum += y;
    } else if (i % 2 === 0) {
      sum += 2 * y;
    } else {
      sum += 4 * y;
    }
  }
  
  return (h / 3) * sum;
}

// Find roots using bisection method
function findRootBisection(
  expr: string,
  a: number,
  b: number,
  params: Record<string, number>,
  exprValues: Record<string, number> = {},
  tolerance: number = 1e-6,
  maxIterations: number = 100
): number | null {
  let fa = evaluateExpression(expr, a, params, exprValues);
  let fb = evaluateExpression(expr, b, params, exprValues);
  
  if (fa === null || fb === null) return null;
  
  if (Math.abs(fa) < tolerance) return a;
  if (Math.abs(fb) < tolerance) return b;
  if (fa * fb > 0) return null; // No root in interval
  
  for (let i = 0; i < maxIterations; i++) {
    const c = (a + b) / 2;
    const fc = evaluateExpression(expr, c, params, exprValues);
    
    if (fc === null) return null;
    
    if (Math.abs(fc) < tolerance || (b - a) / 2 < tolerance) {
      return c;
    }
    
    if (fa * fc < 0) {
      b = c;
      fb = fc;
    } else {
      a = c;
      fa = fc;
    }
  }
  
  return null;
}

// Find all roots in a range
export function findRoots(
  expr: string,
  xMin: number,
  xMax: number,
  params: Record<string, number>,
  exprValues: Record<string, number> = {},
  step: number = 0.1
): number[] {
  const roots: number[] = [];
  const tolerance = 1e-5;
  
  for (let x = xMin; x < xMax; x += step) {
    const y1 = evaluateExpression(expr, x, params, exprValues);
    const y2 = evaluateExpression(expr, x + step, params, exprValues);
    
    if (y1 === null || y2 === null) continue;
    
    // Sign change indicates a root
    if (y1 * y2 <= 0) {
      const root = findRootBisection(expr, x, x + step, params, exprValues, tolerance);
      if (root !== null && !roots.some(r => Math.abs(r - root) < tolerance)) {
        roots.push(root);
      }
    }
  }
  
  return roots;
}

// Find intersection points between two expressions
export function findIntersections(
  expr1: string,
  expr2: string,
  xMin: number,
  xMax: number,
  params: Record<string, number>,
  exprValues1: Record<string, number> = {},
  exprValues2: Record<string, number> = {},
  step: number = 0.1
): Array<{ x: number; y: number }> {
  const intersections: Array<{ x: number; y: number }> = [];
  const tolerance = 1e-5;
  
  for (let x = xMin; x < xMax; x += step) {
    const y1 = evaluateExpression(expr1, x, params, exprValues1);
    const y2 = evaluateExpression(expr2, x, params, exprValues2);
    const y1Next = evaluateExpression(expr1, x + step, params, exprValues1);
    const y2Next = evaluateExpression(expr2, x + step, params, exprValues2);
    
    if (y1 === null || y2 === null || y1Next === null || y2Next === null) continue;
    
    const diff1 = y1 - y2;
    const diff2 = y1Next - y2Next;
    
    // Sign change indicates intersection
    if (diff1 * diff2 <= 0) {
      // Use bisection to find precise intersection
      let a = x;
      let b = x + step;
      
      for (let iter = 0; iter < 50; iter++) {
        const mid = (a + b) / 2;
        const y1Mid = evaluateExpression(expr1, mid, params, exprValues1);
        const y2Mid = evaluateExpression(expr2, mid, params, exprValues2);
        
        if (y1Mid === null || y2Mid === null) break;
        
        const diff = y1Mid - y2Mid;
        
        if (Math.abs(diff) < tolerance || (b - a) < tolerance) {
          intersections.push({ x: mid, y: y1Mid });
          break;
        }
        
        if (diff1 * diff < 0) {
          b = mid;
        } else {
          a = mid;
        }
      }
    }
  }
  
  // Remove duplicates
  return intersections.filter((p, i, arr) => 
    arr.findIndex(q => Math.abs(q.x - p.x) < tolerance) === i
  );
}

// Find local min/max using derivative
export function findMinMax(
  expr: string,
  xMin: number,
  xMax: number,
  params: Record<string, number>,
  exprValues: Record<string, number> = {},
  step: number = 0.1
): Array<{ x: number; y: number; type: 'min' | 'max' }> {
  const extrema: Array<{ x: number; y: number; type: 'min' | 'max' }> = [];
  
  // Find where derivative is zero (critical points)
  const criticalPoints: number[] = [];
  
  for (let x = xMin + step; x < xMax - step; x += step) {
    const deriv1 = calculateDerivative(expr, x, params, exprValues);
    const deriv2 = calculateDerivative(expr, x + step, params, exprValues);
    
    if (deriv1 === null || deriv2 === null) continue;
    
    // Sign change in derivative indicates critical point
    if (deriv1 * deriv2 <= 0) {
      // Refine using bisection on derivative
      let a = x;
      let b = x + step;
      
      for (let iter = 0; iter < 30; iter++) {
        const mid = (a + b) / 2;
        const derivMid = calculateDerivative(expr, mid, params, exprValues);
        
        if (derivMid === null) break;
        
        if (Math.abs(derivMid) < 1e-5 || (b - a) < 1e-6) {
          criticalPoints.push(mid);
          break;
        }
        
        if (deriv1 * derivMid < 0) {
          b = mid;
        } else {
          a = mid;
        }
      }
    }
  }
  
  // Classify each critical point
  for (const x of criticalPoints) {
    const y = evaluateExpression(expr, x, params, exprValues);
    if (y === null) continue;
    
    // Check second derivative to determine min/max
    const secondDeriv = calculateDerivative(
      expr,
      x,
      params,
      exprValues,
      1e-4
    );
    
    if (secondDeriv === null) continue;
    
    // Use second derivative test
    const derivBefore = calculateDerivative(expr, x - step, params, exprValues);
    const derivAfter = calculateDerivative(expr, x + step, params, exprValues);
    
    if (derivBefore !== null && derivAfter !== null) {
      if (derivBefore < 0 && derivAfter > 0) {
        extrema.push({ x, y, type: 'min' });
      } else if (derivBefore > 0 && derivAfter < 0) {
        extrema.push({ x, y, type: 'max' });
      }
    }
  }
  
  return extrema;
}

// Find critical points (where f'(x) = 0)
export function findCriticalPoints(
  expr: string,
  xMin: number,
  xMax: number,
  params: Record<string, number>,
  exprValues: Record<string, number> = {},
  step: number = 0.1
): Array<{ x: number; y: number; type: 'max' | 'min' | 'inflection' }> {
  const critical: Array<{ x: number; y: number; type: 'max' | 'min' | 'inflection' }> = [];
  
  for (let x = xMin + step; x < xMax - step; x += step) {
    const deriv1 = calculateDerivative(expr, x, params, exprValues);
    const deriv2 = calculateDerivative(expr, x + step, params, exprValues);
    
    if (deriv1 === null || deriv2 === null) continue;
    
    // Sign change in derivative
    if (deriv1 * deriv2 <= 0) {
      // Refine critical point
      let a = x;
      let b = x + step;
      
      for (let iter = 0; iter < 30; iter++) {
        const mid = (a + b) / 2;
        const derivMid = calculateDerivative(expr, mid, params, exprValues);
        
        if (derivMid === null) break;
        
        if (Math.abs(derivMid) < 1e-5 || (b - a) < 1e-6) {
          const y = evaluateExpression(expr, mid, params, exprValues);
          if (y === null) break;
          
          // Determine type using second derivative
          const secondDeriv = calculateDerivative(expr, mid, params, exprValues, 1e-4);
          
          if (secondDeriv !== null) {
            if (Math.abs(secondDeriv) < 1e-3) {
              critical.push({ x: mid, y, type: 'inflection' });
            } else if (secondDeriv > 0) {
              critical.push({ x: mid, y, type: 'min' });
            } else {
              critical.push({ x: mid, y, type: 'max' });
            }
          }
          break;
        }
        
        if (deriv1 * derivMid < 0) {
          b = mid;
        } else {
          a = mid;
        }
      }
    }
  }
  
  // Remove duplicates
  return critical.filter((p, i, arr) => 
    arr.findIndex(q => Math.abs(q.x - p.x) < 1e-4) === i
  );
}


'use client';

import React, { useRef, useEffect, useCallback } from 'react';
import { NetworkConfig } from '@/lib/types';

interface Node {
  layerIndex: number;
  nodeIndex: number;
  x: number;
  y: number;
}

interface NetworkGraphProps {
  config: NetworkConfig;
  currentLayer?: number;
  onNodeClick?: (layerIndex: number, nodeIndex?: number) => void;
  width?: number;
  height?: number;
}

export default function NetworkGraph({
  config,
  currentLayer,
  onNodeClick,
  width = 800,
  height = 500,
}: NetworkGraphProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const nodePositionsRef = useRef<Node[]>([]);

  const layoutNetwork = useCallback(() => {
    const nodes: Node[] = [];
    const layerCount = config.layers.length + 2; // +2 for input and output
    const layerSpacing = width / (layerCount + 1);
    const nodeRadius = 15;

    // Input layer
    const inputSize = config.inputSize;
    const inputSpacing = height / (inputSize + 1);
    for (let i = 0; i < inputSize; i++) {
      nodes.push({
        layerIndex: -1, // -1 for input
        nodeIndex: i,
        x: layerSpacing,
        y: inputSpacing * (i + 1),
      });
    }

    // Hidden layers
    for (let l = 0; l < config.layers.length; l++) {
      const layerSize = config.layers[l].neurons;
      const layerSpacingY = height / (layerSize + 1);
      const x = layerSpacing * (l + 2);
      
      for (let n = 0; n < layerSize; n++) {
        nodes.push({
          layerIndex: l,
          nodeIndex: n,
          x,
          y: layerSpacingY * (n + 1),
        });
      }
    }

    // Output layer
    const outputSize = config.layers[config.layers.length - 1].neurons;
    const outputSpacing = height / (outputSize + 1);
    const outputX = layerSpacing * (layerCount);
    for (let i = 0; i < outputSize; i++) {
      nodes.push({
        layerIndex: config.layers.length, // layers.length for output
        nodeIndex: i,
        x: outputX,
        y: outputSpacing * (i + 1),
      });
    }

    nodePositionsRef.current = nodes;
    return nodes;
  }, [config, width, height]);

  const findNodeAt = useCallback((x: number, y: number, threshold: number = 20): Node | null => {
    for (const node of nodePositionsRef.current) {
      const dist = Math.sqrt((x - node.x) ** 2 + (y - node.y) ** 2);
      if (dist < threshold) {
        return node;
      }
    }
    return null;
  }, []);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, width, height);

    const nodes = layoutNetwork();

    // Draw connections (edges)
    ctx.strokeStyle = '#d1d5db';
    ctx.lineWidth = 1;

    // Draw edges between layers
    for (let i = 0; i < nodes.length; i++) {
      const node = nodes[i];
      
      // Find nodes in next layer
      const nextLayerIndex = node.layerIndex === -1 ? 0 : 
                             node.layerIndex < config.layers.length ? node.layerIndex + 1 : null;
      
      if (nextLayerIndex !== null) {
        const nextLayerNodes = nodes.filter(n => n.layerIndex === nextLayerIndex);
        for (const nextNode of nextLayerNodes) {
          ctx.beginPath();
          ctx.moveTo(node.x, node.y);
          ctx.lineTo(nextNode.x, nextNode.y);
          ctx.stroke();
        }
      }
    }

    // Draw nodes
    for (const node of nodes) {
      const isCurrentLayer = currentLayer !== undefined && node.layerIndex === currentLayer;
      const isInputLayer = node.layerIndex === -1;
      const isOutputLayer = node.layerIndex === config.layers.length;

      // Node circle
      ctx.beginPath();
      ctx.arc(node.x, node.y, 15, 0, 2 * Math.PI);
      
      if (isCurrentLayer) {
        ctx.fillStyle = '#3b82f6';
        ctx.strokeStyle = '#1e40af';
        ctx.lineWidth = 3;
      } else if (isInputLayer) {
        ctx.fillStyle = '#10b981';
        ctx.strokeStyle = '#059669';
        ctx.lineWidth = 2;
      } else if (isOutputLayer) {
        ctx.fillStyle = '#f59e0b';
        ctx.strokeStyle = '#d97706';
        ctx.lineWidth = 2;
      } else {
        ctx.fillStyle = '#e5e7eb';
        ctx.strokeStyle = '#9ca3af';
        ctx.lineWidth = 2;
      }
      
      ctx.fill();
      ctx.stroke();

      // Layer label on first node of each layer
      if (node.nodeIndex === 0) {
        ctx.fillStyle = '#374151';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        
        let label = '';
        if (isInputLayer) {
          label = 'Input';
        } else if (isOutputLayer) {
          label = 'Output';
        } else {
          const layer = config.layers[node.layerIndex];
          label = `L${node.layerIndex + 1} (${layer.activation})`;
        }
        
        ctx.fillText(label, node.x, node.y - 20);
      }
    }
  }, [config, currentLayer, width, height, layoutNetwork]);

  useEffect(() => {
    draw();
  }, [draw]);

  const handleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !onNodeClick) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = width / rect.width;
    const scaleY = height / rect.height;

    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    const node = findNodeAt(x, y);
    if (node) {
      onNodeClick(node.layerIndex, node.nodeIndex);
    }
  }, [onNodeClick, findNodeAt, width, height]);

  return (
    <div className="w-full h-full flex items-center justify-center bg-gray-50 rounded-lg border border-gray-300">
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        onClick={handleClick}
        className="cursor-pointer"
        style={{
          maxWidth: '100%',
          maxHeight: '100%',
        }}
      />
    </div>
  );
}


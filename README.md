# DNN Visualizer

An interactive web application for understanding deep neural networks through real-time visualization.

## Features

- **Interactive Data Points**: Click to add, drag to move, right-click to delete data points
- **Real-time Training**: Watch the network learn and adapt in real-time
- **Two Modes**:
  - **Regression**: Network learns to fit a curve through your data points
  - **Classification**: Network learns to separate points into different classes
- **Customizable Architecture**:
  - Adjust number of layers (1-5)
  - Configure neurons per layer (1-32)
  - Choose activation functions (ReLU, Sigmoid, Tanh, Linear, Softmax)
- **Hyperparameter Control**:
  - Learning rate (0.0001 - 1.0)
  - Training epochs per update (1-100)
- **Live Visualization**:
  - Regression: See the prediction curve adapt to your data
  - Classification: See decision boundaries form in real-time
- **Auto-train Mode**: Continuously train the network as you adjust parameters

## Getting Started

### Prerequisites

- Node.js 18+ and npm

### Installation

```bash
npm install
```

### Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Build

```bash
npm run build
npm start
```

## How to Use

1. **Add Data Points**:
   - Click on the canvas to add data points
   - Drag points to move them
   - Right-click to delete points

2. **Configure Network**:
   - Adjust layers, neurons, and activations in the left panel
   - Set learning rate and epochs

3. **Train**:
   - Click "Train" for manual training
   - Enable "Auto-train" for continuous training
   - Watch the loss decrease and predictions improve

4. **Experiment**:
   - Try different architectures
   - Compare activation functions
   - See how learning rate affects training

## Technical Details

- **Framework**: Next.js 14+ with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Visualization**: HTML5 Canvas + Recharts
- **Neural Network**: Custom TypeScript implementation with:
  - Forward and backward propagation
  - Multiple activation functions
  - MSE and Cross-Entropy loss functions
  - Gradient descent optimization

## Project Structure

```
dnn-visualizer/
├── app/              # Next.js app directory
│   ├── page.tsx      # Main application page
│   └── layout.tsx    # Root layout
├── components/       # React components
│   ├── DataCanvas.tsx
│   ├── ParameterControls.tsx
│   └── Chart.tsx
└── lib/              # Core library
    ├── neural-network.ts
    ├── activations.ts
    ├── losses.ts
    └── types.ts
```

## License

This project is for educational purposes.

declare module 'plotly.js-dist-min' {
  interface Plotly {
    newPlot: (div: HTMLElement, data: any[], layout: any, config?: any) => Promise<void>;
    redraw: (div: HTMLElement) => Promise<void>;
  }

  const Plotly: {
    default: Plotly;
    newPlot: (div: HTMLElement, data: any[], layout: any, config?: any) => Promise<void>;
    redraw: (div: HTMLElement) => Promise<void>;
  };

  export default Plotly;
}


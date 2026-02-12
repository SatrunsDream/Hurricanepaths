/// <reference types="vite/client" />

declare module 'react-plotly.js' {
  import { Component } from 'react';
  const ReactPlot: React.ComponentType<{
    data: unknown[];
    layout?: Record<string, unknown>;
    config?: Record<string, unknown>;
    style?: React.CSSProperties;
    useResizeHandler?: boolean;
  }>;
  export default ReactPlot;
}

declare module 'plotly.js' {
  export type Data = Record<string, unknown>;
  export type Layout = Record<string, unknown>;
}

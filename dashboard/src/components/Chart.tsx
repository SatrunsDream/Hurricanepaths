import ReactPlot from 'react-plotly.js';

interface ChartProps {
  data: Record<string, unknown>[];
  layout?: Record<string, unknown>;
  className?: string;
}

const defaultLayout: Record<string, unknown> = {
  template: 'plotly_white',
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  font: { family: 'DM Sans, system-ui, sans-serif', size: 12 },
  margin: { t: 48, r: 24, b: 48, l: 60 },
  legend: {
    orientation: 'h',
    yanchor: 'bottom',
    y: 1.02,
    xanchor: 'right',
    x: 1,
  },
  hovermode: 'x unified',
};

export function Chart({ data, layout = {}, className = '' }: ChartProps) {
  return (
    <div className={className}>
      <ReactPlot
        data={data}
        layout={{ ...defaultLayout, ...layout }}
        useResizeHandler
        style={{ width: '100%', height: '100%', minHeight: 400 }}
        config={{ responsive: true, displayModeBar: true, displaylogo: false }}
      />
    </div>
  );
}

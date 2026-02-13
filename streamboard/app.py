"""
Streamboard - Hurricane Model Comparison Dashboard

A comprehensive Streamlit dashboard for visualizing and comparing
hurricane track forecasting model performance, featuring Null Model
baseline vs Kalman Filter comparison.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
import os

# Page configuration
st.set_page_config(
    page_title="Hurricane Tracker",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - React dashboard inspired (dark sidebar, clean layout)
# NOTE: No leading indentation - Markdown treats indented blocks as code and displays them as text
st.markdown("""<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css" />
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
}
[data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    text-align: left;
    background: transparent;
    color: #94a3b8;
    border: none;
    padding: 10px 12px;
    border-radius: 8px;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #334155;
    color: white;
}
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}
.main {
    background-color: #f8fafc;
}
.main-header {
    font-size: 2rem;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 0.5rem;
}
.sub-header {
    font-size: 1rem;
    color: #64748b;
    margin-bottom: 1.5rem;
}
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    padding: 1rem 1.25rem !important;
    border-radius: 12px !important;
    border: 1px solid rgba(37, 99, 235, 0.5) !important;
    box-shadow: 0 2px 4px rgba(29, 78, 216, 0.3) !important;
}
[data-testid="stMetric"] *, [data-testid="stMetric"] label, [data-testid="stMetric"] p, [data-testid="stMetric"] div {
    color: white !important;
}
.highlight-box {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    padding: 1.25rem 1.5rem;
    border-radius: 12px;
    color: white !important;
    margin: 1rem 0;
    box-shadow: 0 4px 6px rgba(29, 78, 216, 0.3);
}
.highlight-box h2, .highlight-box p, .highlight-box * {
    color: white !important;
    opacity: 1;
}
.highlight-box h2 {
    font-size: 1rem;
    font-weight: 600;
    margin-top: 0;
    margin-bottom: 0.5rem;
}
.highlight-box p {
    font-size: 0.95rem;
    margin: 0;
}
.js-plotly-plot {
    border-radius: 12px;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>""", unsafe_allow_html=True)

# Color scheme
COLORS = {
    'null_model': '#e74c3c',  # Red
    'kalman_filter': '#3498db',  # Blue
    'background': '#f8f9fa',
    'text': '#2c3e50',
    'accent': '#9b59b6'
}

@st.cache_data
def load_data():
    """Load all CSV files"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    
    data = {}
    files = {
        'summary': 'model_comparison_summary.csv',
        'errors': 'error_distributions.csv',
        'storms': 'storm_performance.csv',
        'comparison': 'model_comparison_detailed.csv',
        'metadata': 'storm_metadata.csv'
    }
    
    for key, filename in files.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            data[key] = pd.read_csv(filepath)
            # Convert datetime columns
            if 'start_time' in data[key].columns:
                data[key]['start_time'] = pd.to_datetime(data[key]['start_time'], errors='coerce')
            if 'end_time' in data[key].columns:
                data[key]['end_time'] = pd.to_datetime(data[key]['end_time'], errors='coerce')
        else:
            st.error(f"Missing data file: {filename} in {data_dir}")
            return None
    
    return data

@st.cache_data
def load_track_data():
    """Load full track data for individual storm visualization"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_repo_dir = os.path.dirname(script_dir)
    
    possible_files = [
        os.path.join(main_repo_dir, "data/hurricane_paths_processed.pkl"),
        os.path.join(main_repo_dir, "hurricane_paths_processed.pkl"),
    ]
    
    for filepath in possible_files:
        if os.path.exists(filepath):
            return pd.read_pickle(filepath)
    
    return None

def page_model_comparison_overview(data):
    """Page 1: Model Comparison Overview"""
    st.markdown('<h1 class="main-header">Model Comparison Overview</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">High-level performance comparison: Null Model vs Kalman Filter</p>', unsafe_allow_html=True)
    
    summary = data['summary']
    
    # Key finding highlight
    st.markdown("""
    <div class="highlight-box">
        <h2><i class="fa-solid fa-lightbulb" style="margin-right: 0.5rem;"></i>Key Finding</h2>
        <p>
            The <strong>Null Model (simple persistence)</strong> consistently outperforms the 
            <strong>Kalman Filter</strong> at all lead times, demonstrating that simpler 
            approaches can sometimes achieve better results than sophisticated state-space models.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Summary metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_rmse_kf = summary['kf_rmse'].mean()
        st.metric("Kalman Filter\nAverage RMSE", f"{avg_rmse_kf:.1f} km")
    
    with col2:
        avg_rmse_null = summary['null_rmse'].mean()
        st.metric("Null Model\nAverage RMSE", f"{avg_rmse_null:.1f} km")
    
    with col3:
        improvement = ((avg_rmse_kf - avg_rmse_null) / avg_rmse_kf) * 100
        st.metric("Null Model\nImprovement", f"{improvement:.1f}%", delta=f"{improvement:.1f}% better")
    
    with col4:
        total_forecasts = summary['kf_count'].sum()
        st.metric("Total Forecast\nInstances", f"{total_forecasts:,}")
    
    st.divider()
    
    # RMSE Comparison Line Chart
    fig_rmse = go.Figure()
    
    fig_rmse.add_trace(go.Scatter(
        x=summary['lead_time_hours'],
        y=summary['kf_rmse'],
        mode='lines+markers',
        name='Kalman Filter',
        line=dict(color=COLORS['kalman_filter'], width=3),
        marker=dict(size=10, symbol='circle'),
        hovertemplate='<b>Kalman Filter</b><br>Lead Time: %{x}h<br>RMSE: %{y:.2f} km<extra></extra>'
    ))
    
    fig_rmse.add_trace(go.Scatter(
        x=summary['lead_time_hours'],
        y=summary['null_rmse'],
        mode='lines+markers',
        name='Null Model',
        line=dict(color=COLORS['null_model'], width=3),
        marker=dict(size=10, symbol='square'),
        hovertemplate='<b>Null Model</b><br>Lead Time: %{x}h<br>RMSE: %{y:.2f} km<extra></extra>'
    ))
    
    fig_rmse.update_layout(
        title=dict(
            text='RMSE Comparison by Lead Time',
            font=dict(size=24, color=COLORS['text'])
        ),
        xaxis_title='Lead Time (hours)',
        yaxis_title='RMSE (km)',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig_rmse, use_container_width=True)
    
    # Mean Error Comparison Bar Chart
    col1, col2 = st.columns(2)
    
    with col1:
        fig_mean = go.Figure()
        
        fig_mean.add_trace(go.Bar(
            x=summary['lead_time_hours'],
            y=summary['kf_mean_error'],
            name='Kalman Filter',
            marker_color=COLORS['kalman_filter'],
            hovertemplate='<b>Kalman Filter</b><br>Lead Time: %{x}h<br>Mean Error: %{y:.2f} km<extra></extra>'
        ))
        
        fig_mean.add_trace(go.Bar(
            x=summary['lead_time_hours'],
            y=summary['null_mean_error'],
            name='Null Model',
            marker_color=COLORS['null_model'],
            hovertemplate='<b>Null Model</b><br>Lead Time: %{x}h<br>Mean Error: %{y:.2f} km<extra></extra>'
        ))
        
        fig_mean.update_layout(
            title='Mean Error Comparison',
            xaxis_title='Lead Time (hours)',
            yaxis_title='Mean Error (km)',
            barmode='group',
            template='plotly_white',
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_mean, use_container_width=True)
    
    with col2:
        # Improvement percentage chart
        summary['improvement_pct'] = ((summary['kf_rmse'] - summary['null_rmse']) / summary['kf_rmse']) * 100
        
        fig_improve = go.Figure()
        
        fig_improve.add_trace(go.Bar(
            x=summary['lead_time_hours'],
            y=summary['improvement_pct'],
            marker_color=COLORS['null_model'],
            hovertemplate='<b>Null Model Improvement</b><br>Lead Time: %{x}h<br>Improvement: %{y:.2f}%<extra></extra>',
            text=[f"{val:.1f}%" for val in summary['improvement_pct']],
            textposition='outside'
        ))
        
        fig_improve.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig_improve.update_layout(
            title='Null Model Improvement Over Kalman Filter',
            xaxis_title='Lead Time (hours)',
            yaxis_title='Improvement (%)',
            template='plotly_white',
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_improve, use_container_width=True)
    
    # Summary Statistics Table
    st.markdown("### <i class='fa-solid fa-table' style='color: #64748b; margin-right: 0.5rem;'></i>Detailed Performance Metrics", unsafe_allow_html=True)
    
    display_summary = summary.copy()
    display_summary['improvement_pct'] = ((display_summary['kf_rmse'] - display_summary['null_rmse']) / display_summary['kf_rmse']) * 100
    
    # Format columns for display
    for col in ['kf_mean_error', 'kf_rmse', 'null_mean_error', 'null_rmse']:
        display_summary[col] = display_summary[col].round(2)
    
    display_summary['improvement_pct'] = display_summary['improvement_pct'].round(2)
    
    st.dataframe(
        display_summary[[
            'lead_time_hours', 'kf_mean_error', 'kf_rmse', 
            'null_mean_error', 'null_rmse', 'improvement_pct'
        ]].rename(columns={
            'lead_time_hours': 'Lead Time (h)',
            'kf_mean_error': 'KF Mean Error (km)',
            'kf_rmse': 'KF RMSE (km)',
            'null_mean_error': 'Null Mean Error (km)',
            'null_rmse': 'Null RMSE (km)',
            'improvement_pct': 'Improvement (%)'
        }),
        use_container_width=True,
        hide_index=True
    )

def page_error_distributions(data):
    """Page 2: Error Distribution Analysis"""
    st.markdown('<h1 class="main-header">Error Distribution Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Deep dive into forecast error patterns and distributions</p>', unsafe_allow_html=True)
    
    errors_df = data['errors']
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        lead_times = sorted(errors_df['lead_time_hours'].unique())
        selected_lead_time = st.selectbox("Lead Time", lead_times, index=2)  # Default to 24h
    
    with col2:
        models = ['Both', 'Kalman Filter', 'Null Model']
        selected_model = st.selectbox("Model", models)
    
    with col3:
        if 'basin' in errors_df.columns:
            basins = ['All'] + sorted(errors_df['basin'].dropna().unique().tolist())
            selected_basin = st.selectbox("Basin", basins)
        else:
            selected_basin = 'All'
            st.selectbox("Basin", ['All'], disabled=True)
    
    with col4:
        if 'nature' in errors_df.columns:
            storm_types = ['All'] + sorted(errors_df['nature'].dropna().unique().tolist())
            selected_storm_type = st.selectbox("Storm Type", storm_types)
        else:
            selected_storm_type = 'All'
            st.selectbox("Storm Type", ['All'], disabled=True)
    
    # Filter data
    filtered_df = errors_df[errors_df['lead_time_hours'] == selected_lead_time].copy()
    
    if selected_model == 'Kalman Filter':
        filtered_df = filtered_df[filtered_df['model'] == 'Kalman Filter']
    elif selected_model == 'Null Model':
        filtered_df = filtered_df[filtered_df['model'] == 'Null Model']
    
    if selected_basin != 'All' and 'basin' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['basin'] == selected_basin]
    
    if selected_storm_type != 'All' and 'nature' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['nature'] == selected_storm_type]
    
    st.info(f"Showing {len(filtered_df):,} forecast instances for {selected_lead_time}h lead time")
    
    # Get data
    kf_data = filtered_df[filtered_df['model'] == 'Kalman Filter']['error_km'].dropna()
    null_data = filtered_df[filtered_df['model'] == 'Null Model']['error_km'].dropna()
    
    # Error Distribution with KDE (stacked vertically)
    st.markdown(f"### <i class='fa-solid fa-chart-line' style='color: #64748b; margin-right: 0.5rem;'></i>Error Distribution with KDE at {selected_lead_time}h Lead Time", unsafe_allow_html=True)
    
    fig_hist = go.Figure()
    
    # Histogram for Kalman Filter
    fig_hist.add_trace(go.Histogram(
        x=kf_data,
        name='Kalman Filter',
        marker_color=COLORS['kalman_filter'],
        opacity=0.6,
        nbinsx=50,
        histnorm='probability density',
        hovertemplate='<b>Kalman Filter</b><br>Error: %{x:.1f} km<br>Density: %{y:.4f}<extra></extra>'
    ))
    
    # Histogram for Null Model
    fig_hist.add_trace(go.Histogram(
        x=null_data,
        name='Null Model',
        marker_color=COLORS['null_model'],
        opacity=0.6,
        nbinsx=50,
        histnorm='probability density',
        hovertemplate='<b>Null Model</b><br>Error: %{x:.1f} km<br>Density: %{y:.4f}<extra></extra>'
    ))
    
    # Add KDE curves
    if len(kf_data) > 1:
        try:
            kf_kde = gaussian_kde(kf_data)
            x_kf = np.linspace(kf_data.min(), kf_data.max(), 200)
            y_kf = kf_kde(x_kf)
            fig_hist.add_trace(go.Scatter(
                x=x_kf,
                y=y_kf,
                mode='lines',
                name='KF KDE',
                line=dict(color=COLORS['kalman_filter'], width=2.5, dash='dash'),
                hovertemplate='<b>KF KDE</b><br>Error: %{x:.1f} km<br>Density: %{y:.4f}<extra></extra>'
            ))
        except:
            pass  # Skip KDE if calculation fails
    
    if len(null_data) > 1:
        try:
            null_kde = gaussian_kde(null_data)
            x_null = np.linspace(null_data.min(), null_data.max(), 200)
            y_null = null_kde(x_null)
            fig_hist.add_trace(go.Scatter(
                x=x_null,
                y=y_null,
                mode='lines',
                name='Null KDE',
                line=dict(color=COLORS['null_model'], width=2.5, dash='dash'),
                hovertemplate='<b>Null KDE</b><br>Error: %{x:.1f} km<br>Density: %{y:.4f}<extra></extra>'
            ))
        except:
            pass  # Skip KDE if calculation fails
    
    # Add mean and median lines
    if len(kf_data) > 0:
        kf_mean = kf_data.mean()
        kf_median = kf_data.median()
        fig_hist.add_vline(
            x=kf_mean, 
            line_dash="dot", 
            line_color=COLORS['kalman_filter'],
            opacity=0.7,
            annotation_text=f"KF Mean: {kf_mean:.1f} km",
            annotation_position="top right"
        )
        fig_hist.add_vline(
            x=kf_median,
            line_dash="dot",
            line_color=COLORS['kalman_filter'],
            opacity=0.5,
            annotation_text=f"KF Median: {kf_median:.1f} km",
            annotation_position="top"
        )
    
    if len(null_data) > 0:
        null_mean = null_data.mean()
        null_median = null_data.median()
        fig_hist.add_vline(
            x=null_mean,
            line_dash="dot",
            line_color=COLORS['null_model'],
            opacity=0.7,
            annotation_text=f"Null Mean: {null_mean:.1f} km",
            annotation_position="top left"
        )
        fig_hist.add_vline(
            x=null_median,
            line_dash="dot",
            line_color=COLORS['null_model'],
            opacity=0.5,
            annotation_text=f"Null Median: {null_median:.1f} km",
            annotation_position="bottom"
        )
    
    fig_hist.update_layout(
        xaxis_title='Forecast Error (km)',
        yaxis_title='Probability Density',
        barmode='overlay',
        template='plotly_white',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Cumulative distribution (stacked below)
    st.markdown("### <i class='fa-solid fa-chart-area' style='color: #64748b; margin-right: 0.5rem;'></i>Cumulative Error Distribution", unsafe_allow_html=True)
    
    fig_cum = go.Figure()
    
    kf_sorted = np.sort(kf_data)
    null_sorted = np.sort(null_data)
    
    fig_cum.add_trace(go.Scatter(
        x=kf_sorted,
        y=np.arange(1, len(kf_sorted) + 1) / len(kf_sorted) * 100,
        mode='lines',
        name='Kalman Filter',
        line=dict(color=COLORS['kalman_filter'], width=3),
        hovertemplate='<b>Kalman Filter</b><br>Error: %{x:.1f} km<br>Cumulative: %{y:.1f}%<extra></extra>'
    ))
    
    fig_cum.add_trace(go.Scatter(
        x=null_sorted,
        y=np.arange(1, len(null_sorted) + 1) / len(null_sorted) * 100,
        mode='lines',
        name='Null Model',
        line=dict(color=COLORS['null_model'], width=3),
        hovertemplate='<b>Null Model</b><br>Error: %{x:.1f} km<br>Cumulative: %{y:.1f}%<extra></extra>'
    ))
    
    fig_cum.update_layout(
        xaxis_title='Forecast Error (km)',
        yaxis_title='Cumulative Percentage (%)',
        template='plotly_white',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_cum, use_container_width=True)
    
    # Box plots comparison
    st.markdown("### <i class='fa-solid fa-chart-simple' style='color: #64748b; margin-right: 0.5rem;'></i>Error Distribution Comparison (Box Plot)", unsafe_allow_html=True)
    
    # Add toggle for log scale
    use_log_scale = st.checkbox("Use logarithmic scale", value=True, help="Log scale helps visualize wide error ranges")
    
    fig_box = go.Figure()
    
    for model in ['Kalman Filter', 'Null Model']:
        model_data = filtered_df[filtered_df['model'] == model]['error_km'].dropna()
        # Filter out zeros/negatives for log scale if needed
        if use_log_scale:
            model_data = model_data[model_data > 0]
        
        fig_box.add_trace(go.Box(
            y=model_data,
            name=model,
            marker_color=COLORS['null_model'] if model == 'Null Model' else COLORS['kalman_filter'],
            boxmean='sd',
            hovertemplate='<b>%{fullData.name}</b><br>Error: %{y:.1f} km<extra></extra>'
        ))
    
    yaxis_title = 'Forecast Error (km)' if not use_log_scale else 'Forecast Error (km, log scale)'
    
    fig_box.update_layout(
        yaxis_title=yaxis_title,
        template='plotly_white',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Apply log scale if requested
    if use_log_scale:
        fig_box.update_yaxes(type="log")
    
    st.plotly_chart(fig_box, use_container_width=True)
    
    # Error category breakdown
    st.markdown("### <i class='fa-solid fa-chart-pie' style='color: #64748b; margin-right: 0.5rem;'></i>Error Category Breakdown", unsafe_allow_html=True)
    
    category_counts = filtered_df.groupby(['model', 'error_category']).size().reset_index(name='count')
    category_pct = filtered_df.groupby(['model', 'error_category']).size().reset_index(name='count')
    category_pct['percentage'] = category_pct.groupby('model')['count'].transform(lambda x: x / x.sum() * 100)
    
    fig_cat = px.bar(
        category_pct,
        x='error_category',
        y='percentage',
        color='model',
        barmode='group',
        color_discrete_map={'Kalman Filter': COLORS['kalman_filter'], 'Null Model': COLORS['null_model']},
        labels={'error_category': 'Error Category', 'percentage': 'Percentage (%)', 'model': 'Model'},
        title='Error Category Distribution'
    )
    
    fig_cat.update_layout(template='plotly_white', height=400)
    st.plotly_chart(fig_cat, use_container_width=True)

def page_storm_performance(data):
    """Page 3: Storm-Level Performance"""
    st.markdown('<h1 class="main-header">Storm-Level Performance</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Explore which storms each model handles better</p>', unsafe_allow_html=True)
    
    storms_df = data['storms']
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        basins = ['All'] + sorted(storms_df['basin'].dropna().unique().tolist())
        selected_basin = st.selectbox("Basin", basins, key='storm_basin')
    
    with col2:
        storm_types = ['All'] + sorted(storms_df['nature'].dropna().unique().tolist())
        selected_storm_type = st.selectbox("Storm Type", storm_types, key='storm_type')
    
    with col3:
        min_error = float(storms_df['kf_mean_error'].min())
        max_error = float(storms_df['kf_mean_error'].max())
        error_threshold = st.slider("Max KF Error (km)", min_error, max_error, max_error)
    
    # Filter data
    filtered_storms = storms_df.copy()
    
    if selected_basin != 'All':
        filtered_storms = filtered_storms[filtered_storms['basin'] == selected_basin]
    
    if selected_storm_type != 'All':
        filtered_storms = filtered_storms[filtered_storms['nature'] == selected_storm_type]
    
    filtered_storms = filtered_storms[filtered_storms['kf_mean_error'] <= error_threshold]
    
    st.info(f"Showing {len(filtered_storms):,} storms")
    
    # Scatter plot: KF error vs Null error
    fig_scatter = go.Figure()
    
    # Color by better model
    kf_better = filtered_storms[filtered_storms['better_model'] == 'Kalman Filter']
    null_better = filtered_storms[filtered_storms['better_model'] == 'Null Model']
    
    fig_scatter.add_trace(go.Scatter(
        x=kf_better['kf_mean_error'],
        y=kf_better['null_mean_error'],
        mode='markers',
        name='KF Better',
        marker=dict(color=COLORS['kalman_filter'], size=6, opacity=0.6),
        hovertemplate='<b>Storm: %{text}</b><br>KF Error: %{x:.1f} km<br>Null Error: %{y:.1f} km<extra></extra>',
        text=kf_better['sid']
    ))
    
    fig_scatter.add_trace(go.Scatter(
        x=null_better['kf_mean_error'],
        y=null_better['null_mean_error'],
        mode='markers',
        name='Null Better',
        marker=dict(color=COLORS['null_model'], size=6, opacity=0.6),
        hovertemplate='<b>Storm: %{text}</b><br>KF Error: %{x:.1f} km<br>Null Error: %{y:.1f} km<extra></extra>',
        text=null_better['sid']
    ))
    
    # Add diagonal reference line
    max_error = max(filtered_storms['kf_mean_error'].max(), filtered_storms['null_mean_error'].max())
    fig_scatter.add_trace(go.Scatter(
        x=[0, max_error],
        y=[0, max_error],
        mode='lines',
        name='Equal Performance',
        line=dict(color='gray', dash='dash', width=2),
        showlegend=True
    ))
    
    fig_scatter.update_layout(
        title='Storm Performance Comparison',
        xaxis_title='Kalman Filter Mean Error (km)',
        yaxis_title='Null Model Mean Error (km)',
        template='plotly_white',
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Better model distribution
    col1, col2 = st.columns(2)
    
    with col1:
        better_model_counts = filtered_storms['better_model'].value_counts()
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=better_model_counts.index,
            values=better_model_counts.values,
            marker_colors=[COLORS['null_model'] if 'Null' in label else COLORS['kalman_filter'] 
                          for label in better_model_counts.index],
            hole=0.4
        )])
        
        fig_pie.update_layout(
            title='Which Model Performs Better?',
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Performance by basin
        if 'basin' in filtered_storms.columns:
            basin_perf = filtered_storms.groupby('basin').agg({
                'kf_mean_error': 'mean',
                'null_mean_error': 'mean'
            }).reset_index()
            
            fig_basin = go.Figure()
            
            fig_basin.add_trace(go.Bar(
                x=basin_perf['basin'],
                y=basin_perf['kf_mean_error'],
                name='Kalman Filter',
                marker_color=COLORS['kalman_filter']
            ))
            
            fig_basin.add_trace(go.Bar(
                x=basin_perf['basin'],
                y=basin_perf['null_mean_error'],
                name='Null Model',
                marker_color=COLORS['null_model']
            ))
            
            fig_basin.update_layout(
                title='Average Error by Basin',
                xaxis_title='Basin',
                yaxis_title='Mean Error (km)',
                barmode='group',
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig_basin, use_container_width=True)
    
    # Storm ranking table
    st.markdown("### <i class='fa-solid fa-ranking-star' style='color: #64748b; margin-right: 0.5rem;'></i>Storm Performance Ranking", unsafe_allow_html=True)
    
    display_storms = filtered_storms.nlargest(20, 'kf_mean_error')[
        ['sid', 'basin', 'nature', 'kf_mean_error', 'null_mean_error', 'better_model']
    ].copy()
    
    display_storms['kf_mean_error'] = display_storms['kf_mean_error'].round(2)
    display_storms['null_mean_error'] = display_storms['null_mean_error'].round(2)
    
    st.dataframe(
        display_storms.rename(columns={
            'sid': 'Storm ID',
            'basin': 'Basin',
            'nature': 'Type',
            'kf_mean_error': 'KF Error (km)',
            'null_mean_error': 'Null Error (km)',
            'better_model': 'Better Model'
        }),
        use_container_width=True,
        hide_index=True
    )

def page_individual_storm_tracker(data):
    """Page 4: Individual Storm Tracker"""
    st.markdown('<h1 class="main-header">Individual Storm Tracker</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Visualize specific storm tracks and forecast comparisons</p>', unsafe_allow_html=True)
    
    comparison_df = data['comparison']
    metadata_df = data['metadata']
    
    # Storm selector
    available_storms = sorted(comparison_df['sid'].unique())
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_storm = st.selectbox("Select Storm", available_storms)
    
    with col2:
        storm_meta = metadata_df[metadata_df['sid'] == selected_storm].iloc[0] if len(metadata_df[metadata_df['sid'] == selected_storm]) > 0 else None
        if storm_meta is not None and pd.notna(storm_meta.get('basin_name')):
            st.info(f"**Basin:** {storm_meta.get('basin_name', 'N/A')}")
    
    # Get storm data
    storm_comparison = comparison_df[comparison_df['sid'] == selected_storm].copy()
    
    if len(storm_comparison) == 0:
        st.warning("No forecast data available for this storm.")
        return
    
    # Error metrics by lead time
    error_by_lead = storm_comparison.groupby('lead_time_hours').agg({
        'kf_error_km': 'mean',
        'null_error_km': 'mean'
    }).reset_index()
    
    # Error trend plot
    fig_trend = go.Figure()
    
    fig_trend.add_trace(go.Scatter(
        x=error_by_lead['lead_time_hours'],
        y=error_by_lead['kf_error_km'],
        mode='lines+markers',
        name='Kalman Filter',
        line=dict(color=COLORS['kalman_filter'], width=3),
        marker=dict(size=10)
    ))
    
    fig_trend.add_trace(go.Scatter(
        x=error_by_lead['lead_time_hours'],
        y=error_by_lead['null_error_km'],
        mode='lines+markers',
        name='Null Model',
        line=dict(color=COLORS['null_model'], width=3),
        marker=dict(size=10)
    ))
    
    fig_trend.update_layout(
        title=f'Forecast Error Trend for Storm {selected_storm}',
        xaxis_title='Lead Time (hours)',
        yaxis_title='Mean Error (km)',
        template='plotly_white',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Error metrics table
    st.markdown("### <i class='fa-solid fa-table' style='color: #64748b; margin-right: 0.5rem;'></i>Error Metrics by Lead Time", unsafe_allow_html=True)
    
    display_errors = error_by_lead.copy()
    display_errors['kf_error_km'] = display_errors['kf_error_km'].round(2)
    display_errors['null_error_km'] = display_errors['null_error_km'].round(2)
    display_errors['difference'] = (display_errors['null_error_km'] - display_errors['kf_error_km']).round(2)
    
    st.dataframe(
        display_errors.rename(columns={
            'lead_time_hours': 'Lead Time (h)',
            'kf_error_km': 'KF Error (km)',
            'null_error_km': 'Null Error (km)',
            'difference': 'Difference (km)'
        }),
        use_container_width=True,
        hide_index=True
    )
    
    # Note about map visualization
    st.info("**Note:** Full track visualization with map overlay requires loading the full track dataset. "
            "This feature can be enhanced by loading `hurricane_paths_processed.pkl`.")

def page_methodology(data):
    """Page 6: Methodology & Documentation"""
    import re
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_repo_dir = os.path.dirname(script_dir)
    methodology_path = os.path.join(script_dir, "methodology.md")
    figures_dir = os.path.join(main_repo_dir, "figures")
    report_path = os.path.join(script_dir, "report.pdf")
    if not os.path.exists(report_path):
        report_path = os.path.join(main_repo_dir, "Report", "Sequential_Bayesian_Inference_Applied_to_Hurricane_Tracking.pdf")
    
    st.markdown('<h1 class="main-header">Methodology & Documentation</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Project overview, model implementations, and evaluation framework</p>', unsafe_allow_html=True)
    
    if os.path.exists(report_path):
        with open(report_path, "rb") as f:
            pdf_bytes = f.read()
        st.download_button("Download Report", data=pdf_bytes, file_name="Sequential_Bayesian_Inference_Applied_to_Hurricane_Tracking.pdf", mime="application/pdf", key="download_report")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Methodology content from methodology.md
    if not os.path.exists(methodology_path):
        st.error("methodology.md not found.")
        return
    with open(methodology_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    i = 0
    current_text_block = []
    while i < len(lines):
        line = lines[i]
        if '![' in line and 'figures/' in line:
            if current_text_block:
                st.markdown(''.join(current_text_block), unsafe_allow_html=True)
                current_text_block = []
            match = re.search(r'!\[([^\]]*)\]\(figures/([^)]+)\)', line)
            if match:
                caption = match.group(1)
                fig_name = match.group(2)
                fig_path = os.path.join(figures_dir, fig_name)
                if os.path.exists(fig_path):
                    st.image(fig_path, caption=caption if caption else None, use_container_width=True)
                else:
                    st.warning(f"Figure not found: {fig_name}")
            if i + 1 < len(lines) and lines[i + 1].strip().startswith('*'):
                caption_line = lines[i + 1].strip().strip('*').strip()
                if caption_line:
                    st.caption(caption_line)
                i += 2
            else:
                i += 1
            continue
        current_text_block.append(line)
        i += 1
    if current_text_block:
        st.markdown(''.join(current_text_block), unsafe_allow_html=True)

def main():
    """Main dashboard application"""
    # Load data
    data = load_data()
    
    if data is None:
        st.error("Failed to load data files. Please run `python preprocess_streamboard.py` first.")
        return
    
    # Sidebar - Hurricane Tracker style (React dashboard inspired)
    st.sidebar.markdown("""
    <div style="padding-bottom: 1rem; border-bottom: 1px solid #334155;">
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
            <i class="fa-solid fa-tornado" style="font-size: 1.5rem; color: #818cf8;"></i>
            <span style="font-weight: 700; font-size: 1.25rem; color: white;">Hurricane Tracker</span>
        </div>
        <p style="color: #94a3b8; font-size: 0.875rem; margin: 0;">Model Comparison Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    
    # Nav with buttons and icons
    nav_items = [
        ("⚖️", "Model Comparison", page_model_comparison_overview),
        ("📊", "Error Distributions", page_error_distributions),
        ("🏆", "Storm Performance", page_storm_performance),
        ("🗺️", "Storm Tracker", page_individual_storm_tracker),
        ("📄", "Methodology", page_methodology),
    ]
    pages = {label: fn for _, label, fn in nav_items}
    if "page" not in st.session_state:
        st.session_state.page = "Model Comparison"
    for icon, label, _ in nav_items:
        if st.sidebar.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True):
            st.session_state.page = label
    selected_page = st.session_state.page
    
    # Display selected page
    pages[selected_page](data)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.813rem; padding-top: 0.5rem;">
        <a href="https://personal-site-iota-weld.vercel.app" target="_blank" style="color: #818cf8; text-decoration: none; font-weight: 500;">
            By Sardor Sobirov
        </a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

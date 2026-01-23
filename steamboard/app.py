"""
Steamboard - Hurricane Model Comparison Dashboard

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
    page_title="Steamboard - Hurricane Model Comparison",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
    <style>
    /* Main styling */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 1rem 0;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    
    .info-box {
        background-color: #f8f9fa;
        border-left: 4px solid #2a5298;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    
    .highlight-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

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
        <h2 style="margin-top: 0;">🔍 Key Finding</h2>
        <p style="font-size: 1.1rem; margin-bottom: 0;">
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
    st.subheader("📊 Detailed Performance Metrics")
    
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
    st.subheader(f"📊 Error Distribution with KDE at {selected_lead_time}h Lead Time")
    
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
    st.subheader("📈 Cumulative Error Distribution")
    
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
    st.subheader("📦 Error Distribution Comparison (Box Plot)")
    
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
    st.subheader("📈 Error Category Breakdown")
    
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
    st.subheader("📊 Storm Performance Ranking")
    
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
    st.subheader("📊 Error Metrics by Lead Time")
    
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
    st.info("💡 **Note:** Full track visualization with map overlay requires loading the full track dataset. "
            "This feature can be enhanced by loading `hurricane_paths_processed.pkl`.")

def page_methodology(data):
    """Page 6: Methodology & Documentation"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    methodology_path = os.path.join(script_dir, "methodology.md")
    main_repo_dir = os.path.dirname(script_dir)
    figures_dir = os.path.join(main_repo_dir, "figures")
    
    if not os.path.exists(methodology_path):
        st.error("methodology.md not found. Please ensure the file exists in the steamboard directory.")
        return
    
    # Read markdown file
    with open(methodology_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Process line by line
    import re
    i = 0
    current_text_block = []
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this is a figure reference: ![caption](figures/filename.png)
        if '![' in line and 'figures/' in line:
            # Render accumulated text block first
            if current_text_block:
                st.markdown(''.join(current_text_block), unsafe_allow_html=True)
                current_text_block = []
            
            # Extract figure info
            match = re.search(r'!\[([^\]]*)\]\(figures/([^)]+)\)', line)
            if match:
                caption = match.group(1)
                fig_name = match.group(2)
                fig_path = os.path.join(figures_dir, fig_name)
                
                # Display image
                if os.path.exists(fig_path):
                    st.image(fig_path, caption=caption if caption else None, use_container_width=True)
                else:
                    st.warning(f"Figure not found: {fig_name}")
            
            # Check if next line is a standalone caption (starts with *)
            if i + 1 < len(lines) and lines[i + 1].strip().startswith('*'):
                caption_line = lines[i + 1].strip().strip('*').strip()
                if caption_line:
                    st.caption(caption_line)
                i += 2  # Skip both the figure line and caption line
            else:
                i += 1
            continue
        
        # Regular text line
        current_text_block.append(line)
        i += 1
    
    # Render any remaining text
    if current_text_block:
        st.markdown(''.join(current_text_block), unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    ## Project Overview
    
    This project implements and evaluates probabilistic state-space models for hurricane track forecasting using 
    sequential Bayesian inference. The primary model is a **Kalman Filter** with adaptive process noise, compared 
    against a **Null Model** baseline using simple velocity persistence. The evaluation reveals a surprising finding: 
    the simple persistence baseline consistently outperforms the sophisticated Kalman Filter at all lead times.
    
    **Dataset**: IBTrACS (International Best Track Archive for Climate Stewardship)  
    **Temporal Range**: 1842-2025 (183 years of hurricane track data)  
    **Total Observations**: 721,960 observations across 13,450 storms  
    **Objective**: Implement probabilistic hurricane track forecasting using sequential Bayesian inference
    """)
    
    st.divider()
    
    # Phase 1: Data Exploration and Cleaning
    st.markdown("""
    ## Phase 1: Exploratory Data Analysis and Data Cleaning
    
    ### Objectives
    1. Understand dataset structure and contents
    2. Assess data quality and completeness
    3. Identify key variables for Kalman filter state-space model
    4. Validate temporal structure (6-hour intervals)
    5. Document data characteristics and units
    
    ### Data Loading and Processing
    
    The IBTrACS dataset contains hurricane track data with a two-header format requiring specialized parsing. 
    The data loading function handles:
    - Two-header format normalization
    - Column name standardization
    - Blank value replacement with NaN
    - Type conversions for mixed-type columns
    - Datetime conversion for temporal analysis
    
    ### Dataset Characteristics
    
    - **Size**: 722,040 observations across 13,530 unique storms
    - **Temporal Range**: 1842-10-25 to 2025-11-23
    - **Columns**: 174 total columns
    - **Position Data**: 100% coverage (lat/lon)
    - **Velocity Data**: 99.99% coverage (storm_speed, storm_dir)
    - **Temporal Structure**: Regular 6-hour observation intervals confirmed
    
    ### Data Quality Assessment
    
    Only 80 storms (0.59%) had missing velocity data, all single-observation storms. After filtering to storms 
    with at least 2 observations (required for velocity computation), we retained 13,450 storms (99.4%) with 
    complete data coverage.
    """)
    
    # Add storm density figure if available
    storm_density_path = os.path.join(figures_dir, "fig3_storm_density.png")
    if os.path.exists(storm_density_path):
        st.image(storm_density_path, caption="Figure 3: Storm Density Distribution", use_container_width=True)
    
    st.divider()
    
    # Phase 2: Feature Engineering
    st.markdown("""
    ## Phase 2: Feature Engineering
    
    ### Velocity Computation
    
    Velocity is computed from position differences using haversine distance calculations, handling longitude wrapping 
    at ±180 degrees and accounting for latitude-dependent longitude scaling. The velocity representation is converted 
    to Cartesian components (v_lat, v_lon) measured in degrees per 6-hour interval, suitable for linear Kalman filter 
    operations.
    
    ### State Vector Design
    
    The state vector is represented as **[lat, lon, v_lat, v_lon]** in degrees, which is then converted to metric 
    coordinates **[x_km, y_km, vx_km, vy_km]** for the Kalman filter implementation. This conversion accounts for 
    spherical geometry with longitude velocity adjusted by the cosine of latitude.
    
    ### Advanced Features
    
    **Temporal Features:**
    - Storm age (hours elapsed since first observation)
    - Day of year and month for seasonal patterns
    
    **Motion Features:**
    - Track curvature (measures how sharply storms are turning)
    - Acceleration components (change in velocity over time)
    - Smoothed velocities using 3-point moving averages
    
    **Regime Classifications:**
    - Latitude regime (tropics, subtropics, mid-latitudes)
    - Motion regime (westward, poleward/recurving, low-motion)
    - Storm stage (disturbance, depression, tropical storm, hurricane, extratropical)
    
    **Land Interaction Features:**
    - Distance to land
    - Binary flag for storms within 200 km of land
    - Land gradient (rate of approach to land)
    
    **Beta-Drift Proxy:**
    - Approximates Coriolis-related drift effects
    
    ### Final Processed Dataset
    
    - **Observations**: 721,960 with zero missing values in state variables
    - **Storms**: 13,450 unique storms
    - **Date Range**: 1842 to 2025
    """)
    
    # Add curvature and land interaction figures
    col1, col2 = st.columns(2)
    with col1:
        curvature_path = os.path.join(figures_dir, "fig4_curvature_histogram.png")
        if os.path.exists(curvature_path):
            st.image(curvature_path, caption="Figure 4: Track Curvature Distribution", use_container_width=True)
    
    with col2:
        land_path = os.path.join(figures_dir, "fig5_land_interaction.png")
        if os.path.exists(land_path):
            st.image(land_path, caption="Figure 5: Land Interaction Analysis", use_container_width=True)
    
    st.divider()
    
    # Phase 3: Kalman Filter Implementation
    st.markdown("""
    ## Phase 3: Kalman Filter Implementation
    
    ### State-Space Model Design
    
    The Kalman filter uses a **constant velocity model** with the following specifications:
    
    **State Vector (x_t)**: [x_km, y_km, vx_km, vy_km] in metric coordinates
    - x_km: east-west position in km (relative to storm start)
    - y_km: north-south position in km (relative to storm start)
    - vx_km: east-west velocity in km per 6 hours
    - vy_km: north-south velocity in km per 6 hours
    
    **Observation Vector (y_t)**: [x_km, y_km] - observed positions only
    
    **Transition Matrix (A)**: Implements constant velocity dynamics
    - Position updates linearly with velocity: x_{t+1} = x_t + vx, y_{t+1} = y_t + vy
    - Velocities remain constant: vx_{t+1} = vx_t, vy_{t+1} = vy_t
    
    **Observation Matrix (H)**: Maps state to observations by selecting position components only
    """)
    
    # Add Kalman filter cycle figure
    kf_cycle_path = os.path.join(figures_dir, "fig7_kf_cycle.png")
    if os.path.exists(kf_cycle_path):
        st.image(kf_cycle_path, caption="Figure 7: Kalman Filter Prediction-Update Cycle", use_container_width=True)
    
    st.markdown("""
    ### Parameter Estimation
    
    **Process Noise (Q)**: Estimated from training data by computing the covariance of innovations (differences 
    between predicted and actual state transitions) under the constant velocity model. This captures the inherent 
    uncertainty in storm motion dynamics.
    
    **Observation Noise (R)**: Estimated from observation residuals during filtering. The estimated R has values of 
    approximately 264.92 km² variance for x-coordinate and 168.61 km² variance for y-coordinate, reflecting realistic 
    best-track uncertainty.
    
    ### Feature-Adaptive Process Noise
    
    The filter implements adaptive process noise (Q) scaling based on storm features:
    - **Track curvature**: Higher Q (up to 3×) when storms are turning sharply
    - **Land approach**: 1.5× Q scaling when storms are within 200 km of land
    - **Motion regimes**: Different Q scaling for westward (1.1×), poleward/recurving (1.3×), and low-motion (1.0×) patterns
    - **Latitude regimes**: 1.2× Q scaling in mid-latitudes
    
    ### Train/Test Split
    
    The data is split at the storm level (80/20) to ensure no data leakage:
    - **Training**: 10,759 storms (577,711 observations)
    - **Test**: 2,690 storms (144,247 observations)
    
    ### Evaluation Framework
    
    **Open-Loop Forecasting**: True forecasting evaluation where no future observations are used after initialization.
    
    **Sliding Origin Evaluation**: Forecasts are generated from multiple points along each storm track rather than 
    only from storm origins. This provides robust statistics by sampling diverse storm phases (early development, 
    mature stage, decay, etc.).
    
    For each storm, the function:
    - Identifies valid forecast origins (requiring sufficient history and future data)
    - Samples origins evenly spaced along the track
    - Generates open-loop forecasts from each origin for multiple lead times (6, 12, 24, 48, 72 hours)
    - Aggregates errors across all origins and storms
    """)
    
    # Add forecast error figures
    col1, col2 = st.columns(2)
    with col1:
        forecast_error_path = os.path.join(figures_dir, "fig8_forecast_error.png")
        if os.path.exists(forecast_error_path):
            st.image(forecast_error_path, caption="Figure 8: Forecast Error Analysis", use_container_width=True)
    
    with col2:
        rmse_path = os.path.join(figures_dir, "fig8a_forecast_error_rmse.png")
        if os.path.exists(rmse_path):
            st.image(rmse_path, caption="Figure 8a: RMSE by Lead Time", use_container_width=True)
    
    st.divider()
    
    # Phase 4: Results and Evaluation
    st.markdown("""
    ## Phase 4: Results and Evaluation
    
    ### Kalman Filter Performance
    
    **Open-Loop Forecast Results** (with sliding origins):
    - **6 hours**: Mean error 11.90 km, RMSE 15.86 km
    - **12 hours**: Mean error 24.53 km, RMSE 34.50 km
    - **24 hours**: Mean error 58.13 km, RMSE 83.79 km
    - **48 hours**: Mean error 158.95 km, RMSE 216.74 km
    - **72 hours**: Mean error 286.08 km, RMSE 379.35 km
    
    These results demonstrate expected monotonic error growth with increasing lead time, confirming that open-loop 
    forecasting correctly captures forecast skill rather than filtering accuracy.
    """)
    
    # Add error distribution figures
    col1, col2 = st.columns(2)
    with col1:
        error_dist_path = os.path.join(figures_dir, "fig9_error_distribution.png")
        if os.path.exists(error_dist_path):
            st.image(error_dist_path, caption="Figure 9: Error Distribution Analysis", use_container_width=True)
    
    with col2:
        rmse_dist_path = os.path.join(figures_dir, "fig9a_rmse_distribution.png")
        if os.path.exists(rmse_dist_path):
            st.image(rmse_dist_path, caption="Figure 9a: RMSE Distribution", use_container_width=True)
    
    # Error by lead time
    error_by_lead_path = os.path.join(figures_dir, "fig8b_error_distribution_by_leadtime.png")
    if os.path.exists(error_by_lead_path):
        st.image(error_by_lead_path, caption="Figure 8b: Error Distribution by Lead Time", use_container_width=True)
    
    st.markdown("""
    ### Key Findings from Kalman Filter Evaluation
    
    1. **Short-term accuracy**: 6-hour forecasts achieve approximately 12 km mean error, suitable for operational use
    2. **Error growth pattern**: Errors increase approximately quadratically with lead time, consistent with cumulative 
       process noise in the constant velocity model
    3. **Feature adaptation**: Adaptive Q scaling enables better handling of turning storms and land interactions, 
       though the constant velocity assumption still limits performance for sharp turns
    4. **Observation uncertainty**: Estimated observation noise reflects realistic best-track uncertainty and coordinate 
       conversion errors
    5. **Limitations**: The constant velocity model struggles with rapid motion changes, sharp turns, and extratropical 
       transitions
    """)
    
    # Innovation analysis
    st.markdown("""
    ### Innovation Analysis
    
    The innovation vector (difference between observed and predicted positions) should cluster around zero with no 
    systematic directional bias in a well-behaved Kalman filter. Analysis reveals that the filter behaves consistently 
    with linear-Gaussian assumptions, though adaptive features show weak correlation with forecast uncertainty.
    """)
    
    innovation_path = os.path.join(figures_dir, "fig10_innovation_analysis.png")
    if os.path.exists(innovation_path):
        st.image(innovation_path, caption="Figure 10: Innovation Analysis", use_container_width=True)
    
    st.divider()
    
    # Phase 5: Null Model Baseline
    st.markdown("""
    ## Phase 5: Null Model Baseline Evaluation
    
    ### Null Model Implementation
    
    The null model implements a straightforward **persistence approach** that computes velocity from the difference 
    between the position at the forecast origin time and the position one time step earlier. This velocity is then used 
    to extrapolate the storm's position forward in time without any updates or corrections.
    
    ### Baseline Performance Results
    
    **Test Set Results** (13,892 forecast instances):
    - **6 hours**: Mean error 13.31 km, RMSE 17.77 km
    - **12 hours**: Mean error 29.36 km, RMSE 39.53 km
    - **24 hours**: Mean error 70.56 km, RMSE 95.08 km
    - **48 hours**: Mean error 183.20 km, RMSE 241.14 km
    - **72 hours**: Mean error 326.04 km, RMSE 425.59 km
    
    ### Comparison with Kalman Filter
    
    **Surprising Finding**: The simple persistence baseline **consistently outperforms the Kalman Filter** at all lead times:
    - **6 hours**: Null Model RMSE 17.77 km vs KF 21.70 km (**22.1% improvement**)
    - **12 hours**: Null Model RMSE 39.53 km vs KF 45.75 km (**15.7% improvement**)
    - **24 hours**: Null Model RMSE 95.08 km vs KF 103.90 km (**9.3% improvement**)
    - **48 hours**: Null Model RMSE 241.14 km vs KF 252.31 km (**4.6% improvement**)
    - **72 hours**: Null Model RMSE 425.59 km vs KF 438.76 km (**3.1% improvement**)
    
    This result has profound implications: the Kalman filter's additional complexity may be introducing errors rather 
    than reducing them. Possible explanations include:
    - Suboptimal parameter tuning (process noise Q and observation noise R)
    - Error accumulation through recursive updating
    - The null model's direct velocity persistence may be more effective than the filtered state estimates
    """)
    
    # Add model comparison figures
    model_comp_path = os.path.join(figures_dir, "fig11_model_comparison.png")
    if os.path.exists(model_comp_path):
        st.image(model_comp_path, caption="Figure 11: Model Comparison Overview", use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        rmse_comp_path = os.path.join(figures_dir, "fig11a_rmse_comparison.png")
        if os.path.exists(rmse_comp_path):
            st.image(rmse_comp_path, caption="Figure 11a: RMSE Comparison", use_container_width=True)
    
    with col2:
        error_dist_comp_path = os.path.join(figures_dir, "fig11b_error_distribution_comparison.png")
        if os.path.exists(error_dist_comp_path):
            st.image(error_dist_comp_path, caption="Figure 11b: Error Distribution Comparison", use_container_width=True)
    
    # Null model comparison figures
    null_rmse_path = os.path.join(figures_dir, "fig_null_vs_kf_rmse_comparison.png")
    if os.path.exists(null_rmse_path):
        st.image(null_rmse_path, caption="Null Model vs Kalman Filter RMSE Comparison", use_container_width=True)
    
    null_dist_path = os.path.join(figures_dir, "fig_null_vs_kf_error_distributions.png")
    if os.path.exists(null_dist_path):
        st.image(null_dist_path, caption="Null Model vs Kalman Filter Error Distributions", use_container_width=True)
    
    st.divider()
    
    # Visualizations
    st.markdown("""
    ## Visualization and Analysis
    
    ### Error Trajectories
    
    Error spaghetti plots show forecast position error over time for many storms, revealing that most storms follow 
    similar error growth patterns with errors increasing approximately quadratically with lead time.
    """)
    
    spaghetti_path = os.path.join(figures_dir, "fig15a_error_spaghetti.png")
    if os.path.exists(spaghetti_path):
        st.image(spaghetti_path, caption="Figure 15a: Error Trajectories (Spaghetti Plot)", use_container_width=True)
    
    st.markdown("""
    ### Trajectory Visualization
    
    Trajectory spaghetti plots show true tracks for many storms aligned at a shared forecast origin, illustrating 
    the diversity of hurricane trajectories the models are expected to track.
    """)
    
    traj_spaghetti_path = os.path.join(figures_dir, "fig16_trajectory_spaghetti_24h.png")
    if os.path.exists(traj_spaghetti_path):
        st.image(traj_spaghetti_path, caption="Figure 16: Trajectory Spaghetti Plot (24h)", use_container_width=True)
    
    st.markdown("""
    ### Example Storm Forecasts
    
    Individual storm track visualizations demonstrate how forecasts compare to actual storm tracks, enabling visual 
    assessment of model performance.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        traj1_path = os.path.join(figures_dir, "fig14_trajectory_1852266N1727_24h.png")
        if os.path.exists(traj1_path):
            st.image(traj1_path, caption="Example Storm Track 1 (24h)", use_container_width=True)
    
    with col2:
        traj2_path = os.path.join(figures_dir, "fig14_trajectory_1852278N1429_24h.png")
        if os.path.exists(traj2_path):
            st.image(traj2_path, caption="Example Storm Track 2 (24h)", use_container_width=True)
    
    st.divider()
    
    # Conclusions
    st.markdown("""
    ## Conclusions and Implications
    
    ### Key Findings
    
    1. **Null Model Superiority**: Simple velocity persistence outperforms the sophisticated Kalman Filter at all lead times, 
       demonstrating that simpler approaches can sometimes achieve better results than complex state-space models.
    
    2. **Kalman Filter Limitations**: The recursive updating mechanism may be accumulating small errors that compound over 
       time, while the null model's direct use of the most recent velocity observation avoids this error accumulation.
    
    3. **Parameter Sensitivity**: The Kalman filter's performance is highly sensitive to process noise and observation noise 
       parameters, which may not be optimally tuned despite data-driven estimation.
    
    4. **Feature Adaptation**: Adaptive Q scaling based on storm features shows weak correlation with forecast uncertainty, 
       limiting the benefits of feature-adaptive approaches.
    
    5. **Fundamental Limits**: Both models face fundamental limitations in predicting hurricane motion beyond approximately 
       48 hours, where errors become large regardless of the forecasting method used.
    
    ### Implications for Operational Forecasting
    
    The superior performance of the null model, combined with its computational simplicity, suggests that for operational 
    hurricane track forecasting, persistence-based approaches may be more practical than complex state-space models unless 
    substantial improvements can be demonstrated. The decreasing advantage with lead time indicates that both models face 
    fundamental limitations in predicting hurricane motion beyond approximately 48 hours.
    
    ### Future Directions
    
    - Incorporate acceleration terms into state vector for better handling of motion changes
    - Implement non-linear dynamics models for improved accuracy during sharp turns
    - Enhanced feature engineering to identify truly predictive features for adaptive approaches
    - Comparison with operational forecast models (CLIPER, SHIFOR, NHC guidance)
    """)
    
    st.divider()
    
    # References
    st.markdown("""
    ## References
    
    - **IBTrACS Documentation**: https://www.ncei.noaa.gov/sites/g/files/anmtlf171/files/2025-09/IBTrACS_v04r01_column_documentation.pdf
    - **Bril, G. (1995)**: Forecasting hurricane tracks using the Kalman filter. Environmetrics, 6(1), 7-16.
    - **Project Report**: Sequential Bayesian Inference Applied to Hurricane Tracking
    """)

def main():
    """Main dashboard application"""
    # Load data
    data = load_data()
    
    if data is None:
        st.error("Failed to load data files. Please run `python preprocess_steamboard.py` first.")
        return
    
    # Sidebar navigation
    st.sidebar.title("🌀 Steamboard")
    st.sidebar.markdown("---")
    
    pages = {
        "📊 Model Comparison": page_model_comparison_overview,
        "📈 Error Distributions": page_error_distributions,
        "🌪️ Storm Performance": page_storm_performance,
        "🗺️ Individual Storm Tracker": page_individual_storm_tracker,
        "📖 Methodology": page_methodology
    }
    
    selected_page = st.sidebar.radio("Navigate", list(pages.keys()))
    
    # Display selected page
    pages[selected_page](data)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p><strong>Steamboard</strong></p>
        <p>Hurricane Model Comparison Dashboard</p>
        <p style="margin-top: 1rem;">Built with Streamlit & Plotly</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

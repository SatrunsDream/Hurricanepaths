"""
Hurricane Tracking Dashboard - Interactive Risk Analysis Tool

A Streamlit dashboard for exploring hurricane track forecast errors,
similar to BayesBall-style exploratory analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.figure_factory import create_distplot
import plotly.figure_factory as ff
import os

# Page configuration
st.set_page_config(
    page_title="Hurricane Track Forecast Dashboard",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_dashboard_data():
    """Load preprocessed dashboard data"""
    # Get the directory where this script is located, then go up to main repo
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_repo_dir = os.path.dirname(script_dir)  # Go up one level from dashboard/ to main repo
    
    # Try to find dashboard_data.csv in main repo root
    possible_paths = [
        os.path.join(main_repo_dir, "dashboard_data.csv"),  # Main repo root
        "dashboard_data.csv",  # Fallback: current directory
        os.path.join("dashboard", "dashboard_data.csv")  # Fallback: dashboard folder
    ]
    
    for csv_path in possible_paths:
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                # Convert datetime columns
                if 'start_time' in df.columns:
                    df['start_time'] = pd.to_datetime(df['start_time'])
                if 'end_time' in df.columns:
                    df['end_time'] = pd.to_datetime(df['end_time'])
                return df
            except Exception as e:
                st.warning(f"Error loading {csv_path}: {e}")
                continue
    
    st.error("""
    **Dashboard data file not found!**
    
    Please run the preprocessing script first:
    ```bash
    cd "C:\\Users\\sardo\\OneDrive\\Desktop\\Classes\\CSE150A\\hurricanepaths"
    python dashboard/preprocess_dashboard.py
    ```
    
    This will create `dashboard_data.csv` from your pickle files.
    """)
    return None

@st.cache_data
def load_track_data():
    """Load full track data for individual storm visualization"""
    try:
        # Get the directory where this script is located, then go up to main repo
        script_dir = os.path.dirname(os.path.abspath(__file__))
        main_repo_dir = os.path.dirname(script_dir)  # Go up one level from dashboard/ to main repo
        
        # Try different possible file names
        possible_files = [
            os.path.join(main_repo_dir, "data/hurricane_paths_processed.pkl"),  # Primary path
            os.path.join(main_repo_dir, "data/hurricane_paths_processed_MODEL.pkl"),
            os.path.join(main_repo_dir, "hurricane_paths_processed.pkl"),
            os.path.join(main_repo_dir, "hurricane_paths_processed_MODEL.pkl"),
            "data/hurricane_paths_processed.pkl",  # Fallback: relative paths
            "data/hurricane_paths_processed_MODEL.pkl",
            "hurricane_paths_processed.pkl",
            "hurricane_paths_processed_MODEL.pkl"
        ]
        
        for filepath in possible_files:
            if os.path.exists(filepath):
                return pd.read_pickle(filepath)
        return None
    except Exception as e:
        st.warning(f"Could not load track data: {e}")
        return None

def load_structure_md():
    """Load structure.md content"""
    try:
        # Get the directory where this script is located, then go up to main repo
        script_dir = os.path.dirname(os.path.abspath(__file__))
        main_repo_dir = os.path.dirname(script_dir)  # Go up one level from dashboard/ to main repo
        
        structure_path = os.path.join(main_repo_dir, "structure.md")
        if os.path.exists(structure_path):
            with open(structure_path, "r", encoding="utf-8") as f:
                return f.read()
        # Fallback: try relative path
        with open("structure.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "## Methodology\n\nProject documentation file (structure.md) not found."

# Load data
dashboard_data = load_dashboard_data()
track_data = load_track_data()

# Main header
st.markdown('<p class="main-header">🌀 Hurricane Track Forecast Dashboard</p>', unsafe_allow_html=True)
st.markdown("---")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📊 Scenario Explorer", "🗺️ Individual Storm Tracker", "📖 Methodology"])

# ============================================================================
# TAB 1: SCENARIO EXPLORER (BayesBall-style)
# ============================================================================
with tab1:
    if dashboard_data is None:
        st.stop()
    
    st.header("Forecast Error Distribution Explorer")
    st.markdown("""
    Explore forecast error distributions across different scenarios. 
    Filter by basin, storm type, and lead time to understand model reliability.
    """)
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Basin filter
    available_basins = sorted(dashboard_data['basin_name'].dropna().unique())
    selected_basin = st.sidebar.selectbox(
        "Select Basin",
        options=["All"] + available_basins,
        index=0
    )
    
    # Storm type filter
    available_types = sorted(dashboard_data['storm_type'].dropna().unique())
    selected_type = st.sidebar.selectbox(
        "Select Storm Type",
        options=["All"] + available_types,
        index=0
    )
    
    # Lead time selector
    available_lead_times = sorted(dashboard_data['lead_time_hours'].unique())
    selected_lead_time = st.sidebar.selectbox(
        "Select Lead Time",
        options=available_lead_times,
        index=min(2, len(available_lead_times) - 1) if len(available_lead_times) > 2 else 0  # Default to 24h if available
    )
    
    # Apply filters
    filtered_data = dashboard_data.copy()
    
    if selected_basin != "All":
        filtered_data = filtered_data[filtered_data['basin_name'] == selected_basin]
    
    if selected_type != "All":
        filtered_data = filtered_data[filtered_data['storm_type'] == selected_type]
    
    filtered_data = filtered_data[filtered_data['lead_time_hours'] == selected_lead_time]
    
    # Display filtered data info
    st.info(f"**Filtered Results:** {len(filtered_data):,} forecast instances")
    
    if len(filtered_data) == 0:
        st.warning("No data matches the selected filters. Please adjust your filters.")
        st.stop()
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        mean_error = filtered_data['error_km'].mean()
        st.metric("Mean Error", f"{mean_error:.1f} km")
    
    with col2:
        median_error = filtered_data['error_km'].median()
        st.metric("Median Error", f"{median_error:.1f} km")
    
    with col3:
        rmse = np.sqrt(np.mean(filtered_data['error_km']**2))
        st.metric("RMSE", f"{rmse:.1f} km")
    
    with col4:
        max_error = filtered_data['error_km'].max()
        st.metric("Max Outlier", f"{max_error:.1f} km")
    
    st.markdown("---")
    
    # Main visualization: Error distribution
    st.subheader(f"Forecast Error Distribution ({selected_lead_time}h lead time)")
    
    # Create distribution plot
    errors = filtered_data['error_km'].values
    
    # Create histogram with KDE overlay
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=errors,
        nbinsx=50,
        name='Error Distribution',
        marker_color='steelblue',
        opacity=0.7,
        histnorm='probability density'
    ))
    
    # Add KDE curve (approximate with normal distribution for visualization)
    mean_err = np.mean(errors)
    std_err = np.std(errors)
    x_kde = np.linspace(errors.min(), errors.max(), 200)
    y_kde = np.exp(-0.5 * ((x_kde - mean_err) / std_err)**2) / (std_err * np.sqrt(2 * np.pi))
    
    fig.add_trace(go.Scatter(
        x=x_kde,
        y=y_kde,
        mode='lines',
        name='Normal Approximation',
        line=dict(color='crimson', width=2, dash='dash')
    ))
    
    # Add vertical lines for mean and median
    fig.add_vline(x=mean_error, line_dash="dash", line_color="green", 
                  annotation_text=f"Mean: {mean_error:.1f} km", annotation_position="top")
    fig.add_vline(x=median_error, line_dash="dash", line_color="orange",
                  annotation_text=f"Median: {median_error:.1f} km", annotation_position="top")
    
    fig.update_layout(
        title=f"Error Distribution: {selected_basin} | {selected_type} | {selected_lead_time}h",
        xaxis_title="Forecast Error (km)",
        yaxis_title="Density",
        height=500,
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation box
    st.markdown("### 📈 Interpretation")
    
    # Calculate distribution characteristics
    std_dev = np.std(errors)
    cv = std_dev / mean_error if mean_error > 0 else 0  # Coefficient of variation
    
    if cv < 0.5:
        reliability = "**High Precision** - Narrow distribution indicates consistent forecast accuracy"
        color = "green"
    elif cv < 1.0:
        reliability = "**Moderate Precision** - Moderate spread indicates some forecast variability"
        color = "orange"
    else:
        reliability = "**Low Precision** - Wide distribution indicates high forecast uncertainty"
        color = "red"
    
    st.markdown(f"""
    <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid {color};">
    <strong>Reliability Assessment:</strong> {reliability}<br>
    <strong>Standard Deviation:</strong> {std_dev:.1f} km<br>
    <strong>Coefficient of Variation:</strong> {cv:.2f}
    </div>
    """, unsafe_allow_html=True)
    
    # Additional statistics
    with st.expander("📊 Detailed Statistics"):
        st.write(filtered_data['error_km'].describe())
        
        # Percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        st.write("\n**Percentiles:**")
        for p in percentiles:
            val = np.percentile(errors, p)
            st.write(f"{p}th percentile: {val:.1f} km")

# ============================================================================
# TAB 2: INDIVIDUAL STORM TRACKER
# ============================================================================
with tab2:
    st.header("Individual Storm Track Visualization")
    st.markdown("Select a specific storm to view its track and forecast errors.")
    
    if dashboard_data is None:
        st.error("Dashboard data not available. Please run preprocessing first.")
        st.stop()
    
    # Get unique storms
    unique_storms = sorted(dashboard_data['sid'].dropna().unique())
    
    if len(unique_storms) == 0:
        st.warning("No storm data available.")
        st.stop()
    
    # Storm selector
    selected_storm_id = st.selectbox(
        "Select Storm ID",
        options=unique_storms,
        index=0
    )
    
    # Get storm metadata - get first row with non-null metadata
    storm_rows = dashboard_data[dashboard_data['sid'] == selected_storm_id]
    
    # Try to get a row with metadata, otherwise use first row
    storm_meta = None
    for idx, row in storm_rows.iterrows():
        if pd.notna(row.get('basin')) or pd.notna(row.get('basin_name')):
            storm_meta = row
            break
    
    if storm_meta is None:
        storm_meta = storm_rows.iloc[0]
    
    # Helper function to safely get values
    def safe_get(row, key, default='N/A'):
        val = row.get(key, default)
        if pd.isna(val):
            return default
        return val
    
    # Display storm info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        basin_val = safe_get(storm_meta, 'basin_name') or safe_get(storm_meta, 'basin', 'N/A')
        st.metric("Basin", basin_val)
    with col2:
        storm_type_val = safe_get(storm_meta, 'storm_type') or safe_get(storm_meta, 'nature', 'N/A')
        st.metric("Storm Type", storm_type_val)
    with col3:
        season_val = safe_get(storm_meta, 'season', 'N/A')
        if season_val != 'N/A' and pd.notna(season_val):
            try:
                st.metric("Season", int(float(season_val)))
            except:
                st.metric("Season", season_val)
        else:
            st.metric("Season", 'N/A')
    with col4:
        duration = safe_get(storm_meta, 'duration_days', 'N/A')
        if duration != 'N/A' and pd.notna(duration):
            try:
                st.metric("Duration", f"{float(duration):.1f} days")
            except:
                st.metric("Duration", duration)
        else:
            st.metric("Duration", 'N/A')
    
    st.markdown("---")
    
    # Load track data if available
    if track_data is not None:
        storm_track = track_data[track_data['sid'] == selected_storm_id].sort_values('iso_time')
        
        if len(storm_track) > 0:
            # Map visualization
            st.subheader("Storm Track Map")
            
            fig_map = go.Figure()
            
            # Plot actual track
            fig_map.add_trace(go.Scattermapbox(
                lat=storm_track['lat'].values,
                lon=storm_track['lon'].values,
                mode='lines+markers',
                name='Actual Track',
                line=dict(color='blue', width=3),
                marker=dict(size=6, color='blue'),
                text=[f"Time: {t}" for t in storm_track['iso_time'].values],
                hovertemplate='<b>Actual Position</b><br>Lat: %{lat:.2f}<br>Lon: %{lon:.2f}<extra></extra>'
            ))
            
            # Mark start and end
            fig_map.add_trace(go.Scattermapbox(
                lat=[storm_track['lat'].iloc[0]],
                lon=[storm_track['lon'].iloc[0]],
                mode='markers',
                name='Start',
                marker=dict(size=15, color='green', symbol='circle'),
                hovertemplate='<b>Start</b><br>Lat: %{lat:.2f}<br>Lon: %{lon:.2f}<extra></extra>'
            ))
            
            fig_map.add_trace(go.Scattermapbox(
                lat=[storm_track['lat'].iloc[-1]],
                lon=[storm_track['lon'].iloc[-1]],
                mode='markers',
                name='End',
                marker=dict(size=15, color='red', symbol='x'),
                hovertemplate='<b>End</b><br>Lat: %{lat:.2f}<br>Lon: %{lon:.2f}<extra></extra>'
            ))
            
            fig_map.update_layout(
                mapbox=dict(
                    style="open-street-map",
                    center=dict(
                        lat=storm_track['lat'].mean(),
                        lon=storm_track['lon'].mean()
                    ),
                    zoom=4
                ),
                height=600,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.warning(f"No track data found for storm {selected_storm_id}")
    else:
        st.info("Full track data not available. Install track data file to see map visualization.")
    
    # Error metrics for this storm
    st.subheader("Forecast Error Metrics")
    storm_errors = dashboard_data[dashboard_data['sid'] == selected_storm_id]
    
    if len(storm_errors) > 0:
        # Error by lead time
        error_by_leadtime = storm_errors.groupby('lead_time_hours')['error_km'].agg(['mean', 'std', 'min', 'max']).reset_index()
        error_by_leadtime.columns = ['Lead Time (h)', 'Mean Error (km)', 'Std Dev (km)', 'Min Error (km)', 'Max Error (km)']
        
        st.dataframe(error_by_leadtime, use_container_width=True)
        
        # Error trend plot
        fig_errors = go.Figure()
        
        fig_errors.add_trace(go.Scatter(
            x=error_by_leadtime['Lead Time (h)'],
            y=error_by_leadtime['Mean Error (km)'],
            mode='lines+markers',
            name='Mean Error',
            line=dict(color='steelblue', width=3),
            marker=dict(size=10)
        ))
        
        # Add error bars
        fig_errors.add_trace(go.Scatter(
            x=error_by_leadtime['Lead Time (h)'],
            y=error_by_leadtime['Mean Error (km)'] + error_by_leadtime['Std Dev (km)'],
            mode='lines',
            name='+1 Std Dev',
            line=dict(color='lightblue', width=1, dash='dash'),
            showlegend=False
        ))
        
        fig_errors.add_trace(go.Scatter(
            x=error_by_leadtime['Lead Time (h)'],
            y=error_by_leadtime['Mean Error (km)'] - error_by_leadtime['Std Dev (km)'],
            mode='lines',
            name='-1 Std Dev',
            line=dict(color='lightblue', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(173, 216, 230, 0.3)',
            showlegend=False
        ))
        
        fig_errors.update_layout(
            title="Forecast Error vs Lead Time",
            xaxis_title="Lead Time (hours)",
            yaxis_title="Error (km)",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_errors, use_container_width=True)
    else:
        st.warning(f"No forecast error data available for storm {selected_storm_id}")

# ============================================================================
# TAB 3: METHODOLOGY
# ============================================================================
with tab3:
    st.header("Project Methodology")
    st.markdown("---")
    
    # Load and display structure.md
    structure_content = load_structure_md()
    
    # Render markdown content
    st.markdown(structure_content)
    
    # PDF download button
    st.markdown("---")
    st.subheader("📄 Full Report")
    
    pdf_paths = [
        "Report/CSE150A Report.pdf",
        "Report/Sequential_Bayesian_Inference_Applied_to_Hurricane_Tracking.pdf",
        "CSE150A Report.pdf"
    ]
    
    import os
    pdf_found = False
    for pdf_path in pdf_paths:
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as pdf_file:
                pdf_bytes = pdf_file.read()
                st.download_button(
                    label=f"📥 Download Full Report PDF",
                    data=pdf_bytes,
                    file_name=os.path.basename(pdf_path),
                    mime="application/pdf"
                )
                pdf_found = True
                break
    
    if not pdf_found:
        st.info("PDF report not found. Expected locations:")
        for path in pdf_paths:
            st.text(f"  - {path}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "Hurricane Track Forecast Dashboard | Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)


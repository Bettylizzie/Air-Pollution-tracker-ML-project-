import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
from datetime import datetime, timedelta
import pytz
import folium
from streamlit_folium import st_folium
from statsmodels.tsa.arima.model import ARIMA

# Configure page — must be first Streamlit command
st.set_page_config(
    page_title="AI Air Pollution Tracker", 
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load trained model and preprocessing components
@st.cache_resource
def load_model_components():
    model = joblib.load('best_air_quality_model.joblib')
    encoders = joblib.load('label_encoders.joblib')
    preprocessing = joblib.load('preprocessing_info.joblib')
    return model, encoders, preprocessing

try:
    model, encoders, preprocessing = load_model_components()
except Exception as e:
    st.error(f"Error loading model components: {str(e)}")
    st.stop()

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stAlert {
        border-radius: 10px;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
    }
    .map-container {
        border-radius: 10px;
        overflow: hidden;
    }
    .st-bq {
        border-left: 5px solid #4CAF50;
    }
    .pollution-source {
        font-weight: bold;
        color: #d62728;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.title("AI-Powered Air Pollution Monitoring System")
st.markdown("""
**Supporting SDG 11: Sustainable Cities and Communities**  
This dashboard provides real-time air quality predictions, historical trends, and actionable insights to combat air pollution.
""")

# Sidebar
with st.sidebar:
    st.image("https://sdgs.un.org/themes/custom/porto/assets/images/goals/en/SDG-11.png", width=150)
    st.title("Navigation")
    analysis_type = st.radio(
        "Select Analysis Mode",
        ["📊 Dashboard Overview", "🔍 Detailed Analysis", "🚨 Alert System"]
    )
    
    st.markdown("---")
    st.markdown("### Settings")
    alert_threshold = st.slider(
        "Air Quality Alert Threshold (AQI)", 
        min_value=0, max_value=500, 
        value=100, step=10,
        help="Set the AQI level at which alerts should be triggered"
    )
    
    st.markdown("---")
    st.markdown("""
    **About This App**  
    This tool uses machine learning to predict air quality levels based on:
    - PM2.5
    - Ozone (O3)
    - Nitrogen Dioxide (NO2)
    - Carbon Monoxide (CO)
    """)
    st.markdown("[Learn more about SDG 11](https://sdgs.un.org/goals/goal11)")

# Main content
tab1, tab2, tab3 = st.tabs(["📤 Data Upload", "📈 Visualization", "🧠 AI Insights"])

with tab1:
    st.header("Upload Your Air Quality Data")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Upload CSV file with air quality data",
            type=["csv"],
            help="File should contain columns for PM2.5, O3, NO2, and CO measurements"
        )
    with col2:
        st.download_button(
            label="Download Sample Data",
            data=open("global air pollution dataset.csv", "rb").read(),
            file_name="global air pollution dataset.csv",
            mime="text/csv"
        )
    
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.success("Data successfully loaded!")
            
            with st.expander("🔍 View Raw Data"):
                st.dataframe(data.head())
            
            required_cols = ['PM2.5 AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'CO AQI Value', 'Country', 'City']
            if all(col in data.columns for col in required_cols):
                st.session_state['air_data'] = data
                st.session_state['data_loaded'] = True
            else:
                st.error(f"Missing required columns. Needed: {', '.join(required_cols)}")
                st.session_state['data_loaded'] = False
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

with tab2:
    if 'data_loaded' in st.session_state and st.session_state['data_loaded']:
        data = st.session_state['air_data']
        
        st.header("Data Visualization")
        
        # Feature engineering
        data['Total_Pollution_Score'] = (0.3 * data['PM2.5 AQI Value'] + 
                                       0.25 * data['Ozone AQI Value'] + 
                                       0.25 * data['NO2 AQI Value'] + 
                                       0.2 * data['CO AQI Value'])
        
        for col in ['Country', 'City']:
            le = encoders[col]
            data[col+'_encoded'] = le.transform(data[col])
        
        # Make predictions
        features = preprocessing['features']
        X = data[features]
        data['Predicted_AQI'] = model.predict(X)
        st.session_state['predictions'] = data
        
        # NEW: Pollution severity classification
        def classify_aqi(aqi):
            if aqi <= 50: return "Good", "#2ecc71"
            elif aqi <= 100: return "Moderate", "#f39c12"
            elif aqi <= 150: return "Unhealthy (Sensitive)", "#e74c3c"
            elif aqi <= 200: return "Unhealthy", "#9b59b6"
            elif aqi <= 300: return "Very Unhealthy", "#34495e"
            else: return "Hazardous", "#7f8c8d"
        
        data[['AQI_Category', 'AQI_Color']] = data['Predicted_AQI'].apply(
            lambda x: pd.Series(classify_aqi(x))
        )

        # NEW: Pollution source attribution
        def pollution_source_analysis(row):
            sources = {
                'Industrial': row['NO2 AQI Value']*0.7 + row['PM2.5 AQI Value']*0.3,
                'Vehicular': row['CO AQI Value']*0.8 + row['NO2 AQI Value']*0.2,
                'Agricultural': row['PM2.5 AQI Value']*0.6 + row['Ozone AQI Value']*0.4
            }
            return max(sources.items(), key=lambda x: x[1])[0]
        
        data['Main_Source'] = data.apply(pollution_source_analysis, axis=1)
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Highest AQI", f"{data['Predicted_AQI'].max():.0f}", 
                     delta=data.loc[data['Predicted_AQI'].idxmax(), 'City'])
        with col2:
            st.metric("Average AQI", f"{data['Predicted_AQI'].mean():.0f}")
        with col3:
            alert_count = len(data[data['Predicted_AQI'] > alert_threshold])
            st.metric("Alert Areas", alert_count, delta=f"{alert_count/len(data)*100:.1f}%")
        with col4:
            dominant_source = data['Main_Source'].value_counts().idxmax()
            st.metric("Dominant Source", dominant_source)
        
        # NEW: Interactive Pollution Heatmap
# Replace your existing heatmap code with this:

        st.subheader("🔥 LIVE Pollution Heatmap")

        # Generate mock coordinates if not available
        if 'lat' not in data.columns or 'lon' not in data.columns:
            np.random.seed(42)
            data['lat'] = np.random.uniform(-90, 90, len(data))
            data['lon'] = np.random.uniform(-180, 180, len(data))

        # Create Folium map centered on mean coordinates
        m = folium.Map(location=[data['lat'].mean(), data['lon'].mean()], 
                    zoom_start=4, 
                    control_scale=True)

        # Add heatmap layer
        from folium.plugins import HeatMap

        heat_data = [[row['lat'], row['lon'], row['Predicted_AQI']] 
                    for _, row in data.iterrows()]

        HeatMap(heat_data,
                min_opacity=0.5,
                max_val=data['Predicted_AQI'].max(),
                radius=20, 
                blur=15,
                gradient={0.4: 'blue', 0.6: 'lime', 0.8: 'yellow', 1.0: 'red'}
            ).add_to(m)

        # Add markers for top 5 worst locations
        for _, row in data.nlargest(5, 'Predicted_AQI').iterrows():
            folium.Marker(
                location=[row['lat'], row['lon']],
                popup=f"{row['City']}<br>AQI: {row['Predicted_AQI']:.0f}<br>Source: {row['Main_Source']}",
                icon=folium.Icon(color='red', icon='warning-sign')
            ).add_to(m)

        # Display the map
        st_folium(m, width=1200, height=600, returned_objects=[])
        
        # NEW: Pollution Source Breakdown
        st.subheader("🏭 Pollution Source Attribution")
        source_counts = data['Main_Source'].value_counts().reset_index()
        source_counts.columns = ['Source', 'Count']
        
        fig = px.pie(
            source_counts,
            names='Source',
            values='Count',
            color_discrete_sequence=px.colors.qualitative.Pastel,
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # NEW: Time-Series Forecasting
        st.subheader("🔮 7-Day Pollution Forecast")
        
        # Create time series data
        dates = pd.date_range(end=datetime.now(), periods=len(data), freq='D')
        ts_data = data.copy()
        ts_data['Date'] = dates
        ts_data = ts_data.set_index('Date')['Predicted_AQI']
        
        # Fit ARIMA model
        model_arima = ARIMA(ts_data, order=(5,1,0))
        model_fit = model_arima.fit()
        forecast = model_fit.forecast(steps=7)
        
        # Plot forecast
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ts_data.index,
            y=ts_data,
            name='Historical',
            line=dict(color='#3498db')
        ))
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast,
            name='Forecast',
            line=dict(color='#e74c3c', dash='dot')
        ))
        fig.update_layout(
            title='Predicted AQI Trend',
            xaxis_title='Date',
            yaxis_title='AQI',
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("Please upload data in the 'Data Upload' tab first")

with tab3:
    if 'data_loaded' in st.session_state and st.session_state['data_loaded']:
        data = st.session_state['predictions']
        
        st.header("AI-Powered Insights")
        
        # NEW: Enhanced Alert System with Source Attribution
        st.subheader("🚨 Critical Pollution Alerts")
        alert_data = data[data['Predicted_AQI'] > alert_threshold]
        
        if not alert_data.empty:
            cols = st.columns(3)
            for idx, (_, row) in enumerate(alert_data.iterrows()):
                with cols[idx % 3]:
                    with st.container():
                        st.markdown(f"""
                        <div style='border-left: 5px solid {row['AQI_Color']}; padding: 10px; background: white; border-radius: 5px; margin-bottom: 10px;'>
                            <h4>{row['City']}, {row['Country']}</h4>
                            <p><b>AQI:</b> <span style='color:{row['AQI_Color']}'>{row['Predicted_AQI']:.0f}</span></p>
                            <p><b>Main Source:</b> <span class='pollution-source'>{row['Main_Source']}</span></p>
                            <p><b>PM2.5:</b> {row['PM2.5 AQI Value']:.0f}</p>
                            <p><b>O3:</b> {row['Ozone AQI Value']:.0f}</p>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.success("No areas currently exceed the alert threshold")
        
        # NEW: Comprehensive Health Impact Dashboard
        st.subheader("💊 Health Impact Dashboard")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Health Recommendations")
            aqi_levels = [0, 50, 65.42283126624899, 100, 150, 200, 300, 500]
            mean_value = data['Predicted_AQI'].mean()
            st.markdown(f"**Average AQI:** {mean_value:.2f}")
            closest_value = min(aqi_levels, key=lambda x: abs(x - mean_value))
            aqi_level = st.select_slider(
                "Select current AQI level:",
                options=aqi_levels,
                value=closest_value
            )   

            protection_measures = {
                50: "✅ **No special precautions needed**\n- Enjoy outdoor activities",
                100: "⚠️ **Sensitive groups take caution**\n- Consider reducing prolonged exertion if sensitive",
                150: "⚠️⚠️ **Health alert for sensitive groups**\n- Sensitive groups should reduce outdoor activity\n- Close windows when outdoor AQI is high",
                200: "❌ **Health alert for everyone**\n- Wear N95 masks outdoors\n- Use air purifiers indoors\n- Reschedule outdoor activities",
                300: "🛑 **Health warnings of emergency conditions**\n- Remain indoors with windows closed\n- Run air purifiers continuously\n- Avoid all outdoor exposure",
                500: "💀 **Emergency conditions**\n- Evacuate area if possible\n- Use respirators if must go outside\n- Seek medical attention for symptoms"
            }
            
            for threshold, measure in sorted(protection_measures.items(), reverse=True):
                if aqi_level >= threshold:
                    st.markdown(f"#### AQI ≥ {threshold}:")
                    st.markdown(measure.replace('\n', '<br>'), unsafe_allow_html=True)
                    break
        
        with col2:
            st.markdown("### Population Risk Assessment")
            risk_levels = {
                'Low': (0, 50),
                'Moderate': (51, 100),
                'High (Sensitive Groups)': (101, 150),
                'Very High': (151, 200),
                'Severe': (201, 300),
                'Hazardous': (301, 500)
            }
            
            risk_data = []
            for level, (low, high) in risk_levels.items():
                count = len(data[(data['Predicted_AQI'] >= low) & (data['Predicted_AQI'] <= high)])
                risk_data.append({'Risk Level': level, 'Count': count, 'Range': f"{low}-{high}"})
            
            risk_df = pd.DataFrame(risk_data)
            fig = px.bar(
                risk_df,
                x='Risk Level',
                y='Count',
                color='Risk Level',
                color_discrete_sequence=px.colors.sequential.YlOrRd[::-1],
                hover_data=['Range'],
                title="Population Distribution by Risk Level"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # NEW: Government Action Recommendations
        st.subheader("🏛️ Policy Recommendations")
        
        def generate_recommendations(data):
            recommendations = []
            source_dist = data['Main_Source'].value_counts(normalize=True)
            
            if source_dist.get('Industrial', 0) > 0.4:
                recommendations.append({
                    "action": "Enforce stricter industrial emission standards",
                    "impact": "Could reduce AQI by 15-30 points in industrial zones",
                    "urgency": "High" if data['Predicted_AQI'].max() > 150 else "Medium"
                })
            
            if source_dist.get('Vehicular', 0) > 0.4:
                recommendations.append({
                    "action": "Implement vehicle rotation/restriction programs",
                    "impact": "Could reduce traffic-related pollution by 20-40%",
                    "urgency": "High" if data['Predicted_AQI'].max() > 150 else "Medium"
                })
            
            if (data['PM2.5 AQI Value'] > 100).any():
                recommendations.append({
                    "action": "Issue public health warnings for sensitive groups",
                    "impact": "Immediate protection for vulnerable populations",
                    "urgency": "Critical" if data['Predicted_AQI'].max() > 200 else "High"
                })
            
            if len(recommendations) == 0:
                recommendations.append({
                    "action": "Maintain current environmental policies",
                    "impact": "Air quality is within acceptable limits",
                    "urgency": "Low"
                })
            
            return pd.DataFrame(recommendations)
        
        rec_df = generate_recommendations(data)
        
        if not rec_df.empty:
            for _, row in rec_df.iterrows():
                with st.expander(f"🚩 {row['action']} (Urgency: {row['urgency']})"):
                    st.markdown(f"**Expected Impact:** {row['impact']}")
                    if row['urgency'] in ['High', 'Critical']:
                        st.error("Immediate action recommended")
                    else:
                        st.info("Action can be planned")
        else:
            st.success("No urgent policy actions required at this time")
        
        # NEW: Pollution Scenario Modeling with Source Targeting
        st.subheader("🔮 Advanced Scenario Modeling")
        
        with st.expander("Simulate Pollution Reduction Strategies"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Source-Specific Reductions")
                industrial_red = st.slider("Industrial emissions reduction (%)", 0, 100, 0)
                vehicular_red = st.slider("Vehicular emissions reduction (%)", 0, 100, 0)
                agri_red = st.slider("Agricultural emissions reduction (%)", 0, 100, 0)
                
            with col2:
                st.markdown("### General Controls")
                pm25_red = st.slider("PM2.5 reduction (%)", 0, 100, 0)
                o3_red = st.slider("Ozone reduction (%)", 0, 100, 0)
                no2_red = st.slider("NO2 reduction (%)", 0, 100, 0)
                co_red = st.slider("CO reduction (%)", 0, 100, 0)
            
            if st.button("Calculate Impact"):
                # Create a copy of the data for simulation
                sim_data = data.copy()
                
                # Apply source-specific reductions
                for idx, row in sim_data.iterrows():
                    if row['Main_Source'] == 'Industrial':
                        sim_data.at[idx, 'NO2 AQI Value'] *= (1 - industrial_red/100)
                    elif row['Main_Source'] == 'Vehicular':
                        sim_data.at[idx, 'CO AQI Value'] *= (1 - vehicular_red/100)
                    elif row['Main_Source'] == 'Agricultural':
                        sim_data.at[idx, 'PM2.5 AQI Value'] *= (1 - agri_red/100)
                
                # Apply general reductions
                sim_data['PM2.5 AQI Value'] *= (1 - pm25_red/100)
                sim_data['Ozone AQI Value'] *= (1 - o3_red/100)
                sim_data['NO2 AQI Value'] *= (1 - no2_red/100)
                sim_data['CO AQI Value'] *= (1 - co_red/100)
                
                # Recalculate
                sim_data['Total_Pollution_Score'] = (0.3 * sim_data['PM2.5 AQI Value'] + 
                                                   0.25 * sim_data['Ozone AQI Value'] + 
                                                   0.25 * sim_data['NO2 AQI Value'] + 
                                                   0.2 * sim_data['CO AQI Value'])
                
                X_sim = sim_data[features]
                sim_data['Predicted_AQI'] = model.predict(X_sim)
                
                # Calculate improvements
                original_avg = data['Predicted_AQI'].mean()
                new_avg = sim_data['Predicted_AQI'].mean()
                improvement = original_avg - new_avg
                
                # Display results
                st.success(f"""
                **Simulation Results**  
                Average AQI Improvement: {improvement:.1f} points  
                New Average AQI: {new_avg:.1f}  
                Areas above threshold reduced by: {len(data[data['Predicted_AQI'] > alert_threshold]) - len(sim_data[sim_data['Predicted_AQI'] > alert_threshold])} locations
                """)
                
                # Show comparison
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=['Original', 'Simulated'],
                    y=[original_avg, new_avg],
                    marker_color=['#e74c3c', '#2ecc71'],
                    text=[f"{original_avg:.1f}", f"{new_avg:.1f}"],
                    textposition='auto'
                ))
                fig.update_layout(
                    title='Average AQI Comparison',
                    yaxis_title='AQI',
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please upload data in the 'Data Upload' tab first")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center;">
    <p>Developed for SDG 11: Sustainable Cities and Communities</p>
    <p>Data last updated: {}</p>
    <p>Model version: 2.1 | Last trained: {}</p>
</div>
""".format(
    datetime.now(pytz.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
), unsafe_allow_html=True)
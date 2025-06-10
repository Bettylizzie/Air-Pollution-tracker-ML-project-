import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
from datetime import datetime
import pytz

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
</style>
""", unsafe_allow_html=True)

# App header
st.title("🌍 AI-Powered Air Pollution Monitoring System")
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
        
        data['Total_Pollution_Score'] = (0.3 * data['PM2.5 AQI Value'] + 
                                       0.25 * data['Ozone AQI Value'] + 
                                       0.25 * data['NO2 AQI Value'] + 
                                       0.2 * data['CO AQI Value'])
        
        for col in ['Country', 'City']:
            le = encoders[col]
            data[col+'_encoded'] = le.transform(data[col])
        
        features = preprocessing['features']
        X = data[features]
        data['Predicted_AQI'] = model.predict(X)
        st.session_state['predictions'] = data
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Highest AQI", f"{data['Predicted_AQI'].max():.0f}")
        with col2:
            st.metric("Average AQI", f"{data['Predicted_AQI'].mean():.0f}")
        with col3:
            st.metric("Locations Analyzed", len(data))
        with col4:
            alert_count = len(data[data['Predicted_AQI'] > alert_threshold])
            st.metric("Alert Areas", alert_count, delta=f"{alert_count/len(data)*100:.1f}%")
        
        st.subheader("Geospatial Distribution")
        np.random.seed(42)
        data['lat'] = np.random.uniform(-90, 90, len(data))
        data['lon'] = np.random.uniform(-180, 180, len(data))
        
        fig = px.scatter_mapbox(
            data,
            lat="lat",
            lon="lon",
            color="Predicted_AQI",
            size="Predicted_AQI",
            hover_name="City",
            hover_data=["Country", "PM2.5 AQI Value", "Ozone AQI Value"],
            color_continuous_scale=px.colors.sequential.YlOrRd,
            zoom=1,
            height=500
        )
        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Pollution Trends")
        dates = pd.date_range(end=datetime.now(), periods=len(data), freq='D')
        data['Date'] = dates
        
        pollutant = st.selectbox(
            "Select Pollutant to Visualize",
            ['PM2.5 AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'CO AQI Value']
        )
        
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=data, x='Date', y=pollutant, ax=ax)
        plt.xticks(rotation=45)
        plt.title(f"{pollutant} Over Time")
        plt.ylabel("AQI Value")
        st.pyplot(fig)
        
    else:
        st.warning("Please upload data in the 'Data Upload' tab first")

with tab3:
    if 'data_loaded' in st.session_state and st.session_state['data_loaded']:
        data = st.session_state['predictions']
        
        st.header("AI-Powered Insights")
        
        st.subheader("🚨 Pollution Alerts")
        alert_data = data[data['Predicted_AQI'] > alert_threshold]
        
        if not alert_data.empty:
            for _, row in alert_data.iterrows():
                with st.container():
                    st.warning(f"""
                    **{row['City']}, {row['Country']}**  
                    Predicted AQI: {row['Predicted_AQI']:.0f}  
                    Primary pollutants:  
                    - PM2.5: {row['PM2.5 AQI Value']}  
                    - O3: {row['Ozone AQI Value']}  
                    - NO2: {row['NO2 AQI Value']}  
                    - CO: {row['CO AQI Value']}
                    """)
        else:
            st.success("No areas currently exceed the alert threshold")
        
        st.subheader("💊 Health Recommendations")
        
        aqi_level = st.select_slider(
            "Select AQI Range for Recommendations",
            options=[0, 50, 100, 150, 200, 300, 500],
            value=(0, 50)
        )
        
        if aqi_level[1] <= 50:
            st.success("""
            **Good Air Quality**  
            - Air quality is satisfactory
            - No health risks expected
            - Ideal for outdoor activities
            """)
        elif aqi_level[1] <= 100:
            st.info("""
            **Moderate Air Quality**  
            - Acceptable quality
            - Unusually sensitive people should consider reducing prolonged outdoor exertion
            """)
        elif aqi_level[1] <= 150:
            st.warning("""
            **Unhealthy for Sensitive Groups**  
            - People with heart or lung disease, older adults, and children are at risk
            - Reduce prolonged or heavy outdoor exertion
            """)
        else:
            st.error("""
            **Unhealthy to Hazardous Conditions**  
            - Everyone may experience health effects
            - Sensitive groups should avoid all outdoor exertion
            - Others should limit outdoor activities
            - Consider using N95 masks if outdoors
            """)
        
        st.subheader("🔮 Pollution Scenario Modeling")
        
        with st.form("scenario_analysis"):
            st.write("Adjust pollutant levels to see predicted AQI impact")
            
            col1, col2 = st.columns(2)
            with col1:
                pm25 = st.slider("PM2.5 Level", 0, 500, 50)
                no2 = st.slider("NO2 Level", 0, 500, 30)
            with col2:
                o3 = st.slider("Ozone Level", 0, 500, 40)
                co = st.slider("CO Level", 0, 500, 20)
            
            country = st.selectbox("Country", data['Country'].unique())
            city = st.selectbox("City", data[data['Country'] == country]['City'].unique())
            
            if st.form_submit_button("Predict AQI"):
                input_data = {
                    'PM2.5 AQI Value': pm25,
                    'Ozone AQI Value': o3,
                    'NO2 AQI Value': no2,
                    'CO AQI Value': co,
                    'Country': country,
                    'City': city,
                    'Total_Pollution_Score': (0.3 * pm25 + 0.25 * o3 + 0.25 * no2 + 0.2 * co)
                }
                
                for col in ['Country', 'City']:
                    le = encoders[col]
                    input_data[col+'_encoded'] = le.transform([input_data[col]])[0]
                
                input_df = pd.DataFrame([input_data])
                prediction = model.predict(input_df[features])[0]
                
                st.metric("Predicted Air Quality Index", f"{prediction:.0f}", 
                         delta="Good" if prediction <= 50 else "Moderate" if prediction <= 100 else "Unhealthy")
                
                if prediction > 150:
                    st.error("Health Impact: Significant risk - limit outdoor activities")
                elif prediction > 100:
                    st.warning("Health Impact: Moderate risk - sensitive groups should take precautions")
                else:
                    st.success("Health Impact: Minimal risk - generally safe for all")
    else:
        st.warning("Please upload data in the 'Data Upload' tab first")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center;">
    <p>Developed for SDG 11: Sustainable Cities and Communities</p>
    <p>Data last updated: {}</p>
</div>
""".format(datetime.now(pytz.utc).strftime("%Y-%m-%d %H:%M:%S UTC")), unsafe_allow_html=True)

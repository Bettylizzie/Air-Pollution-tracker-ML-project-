# AI-Powered Air Pollution Tracker

This project is an intelligent, interactive web-based dashboard that leverages **machine learning** to **predict and monitor air quality levels** using real-world environmental data. It supports **Sustainable Development Goal (SDG) 11: Sustainable Cities and Communities** by helping individuals, communities, and policymakers identify, understand, and respond to harmful pollution trends.

## Features

- Predicts Air Quality Index (AQI) using PM2.5, O3, NO2, and CO values
- Visualizes pollution trends and spatial patterns across cities
- Maps pollution intensity using heatmaps and scatter maps
- Alerts users about high-pollution zones
- Provides health advice and “what-if” pollution scenario modeling
- Helps support data-driven decision-making for urban sustainability

---

## Goals

- **Track and predict air quality** in urban and rural areas
- **Empower communities** with accessible air pollution insights
- **Support SDG 11** by highlighting air quality challenges in cities
- **Bridge the gap** between ML technology and environmental monitoring

---

## Importance

Air pollution is a leading environmental health risk, contributing to millions of premature deaths each year. This app:

- Helps individuals avoid high-risk areas
- Assists city planners in identifying pollution hotspots
- Raises awareness about pollution sources and preventive measures

---

##  Gaps & Opportunities

- Currently uses simulated coordinates — could be improved with real geolocation data
- Time-series trends are synthesized — integrating live or historical data sources like OpenAQ or NASA satellites could enhance accuracy
- No API integration yet — potential to integrate with government or environmental APIs for real-time data
- Not yet optimized for mobile — responsive enhancements can make it more accessible

---

## Running the App Locally

> Make sure you have Python 3.8+ and Git installed.

### 1. Clone the Repository
git clone https://github.com/Bettylizzie/Air-Pollution-tracker-ML-project-.git
cd Air-Pollution-tracker-ML-project-


### 2. Create and Activate a Virtual Environment
  # Windows
  python -m venv venv
  venv\Scripts\activate
  
  # macOS/Linux
  python3 -m venv venv
  source venv/bin/activate

### 3. Install Required Packages
  pip install -r requirements.txt

### 4. Add Required Model Files
Ensure these files are in the root folder:

best_air_quality_model.joblib

label_encoders.joblib

preprocessing_info.joblib

global air pollution dataset.csv

(If not present, upload your own or request them from the project owner)

### 5. Run the App
streamlit run app.py

### 6. Contributing
Contributions are welcome! You can:

Submit pull requests

Report bugs via Issues

Suggest enhancements for better performance or UI

### Related Goals
Supports:

SDG 11 – Sustainable Cities and Communities

SDG 13 – Climate Action

SDG 3 – Good Health and Well-being

### Developed By
Betty Njuguna - Data Scientist/AI Engineer
AI for Climate Advocate
GitHub: @Bettylizzie









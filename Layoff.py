import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

st.set_page_config(
    page_title="Employee Layoff Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def add_bg_from_local(image_file):
    try:
        with open(image_file, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode()
            st.markdown(f"""
                <style>
                .stApp {{
                    background-image: url(data:image/png;base64,{img_data});
                    background-size: cover;
                    background-position: center;
                    background-attachment: fixed;
                }}
                .stApp::before {{
                    content: '';
                    position: fixed;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background: rgba(255, 255, 255, 0.7);
                    z-index: -1;
                }}
                </style>
                """, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("⚠️ Background image not found.")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main { background-color: transparent; }
    
    .header-box {
        background: linear-gradient(135deg, rgba(26, 58, 82, 0.95), rgba(45, 90, 123, 0.95));
        padding: 40px;
        border-radius: 20px;
        margin-bottom: 30px;
        border: 2px solid rgba(26, 58, 82, 0.5);
        box-shadow: 0 25px 60px rgba(0, 0, 0, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        animation: slideDown 0.6s ease-out;
    }
    
    @keyframes slideDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .header-box h1 {
        color: #ffffff;
        font-size: 3em;
        font-weight: 800;
        text-shadow: 0 4px 15px rgba(26, 58, 82, 0.4);
        margin: 0;
    }
    
    .subtitle {
        color: #e8f1ff;
        font-size: 1.2em;
        margin-top: 12px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    .section-header-box {
        background: linear-gradient(135deg, rgba(26, 58, 82, 0.85), rgba(45, 90, 123, 0.85));
        padding: 18px 25px;
        border-radius: 15px;
        border: 2px solid rgba(26, 58, 82, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin-bottom: 0;
    }
    
    .section-header-box h3 {
        color: #ffffff;
        font-size: 1.3em;
        font-weight: 800;
        margin: 0;
        text-shadow: 0 2px 8px rgba(26, 58, 82, 0.3);
    }
    
    .glass-card {
        background: rgba(250, 250, 255, 0.85);
        padding: 25px;
        border-radius: 15px;
        border: 2px solid rgba(26, 58, 82, 0.2);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.08);
        backdrop-filter: blur(10px);
        margin: 15px 0;
        transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        position: relative;
        overflow: hidden;
    }
    
    .glass-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.3), transparent);
        transform: rotate(45deg);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    .glass-card:hover {
        border: 2px solid rgba(26, 58, 82, 0.5);
        box-shadow: 0 20px 60px rgba(26, 58, 82, 0.25), 
                    0 0 30px rgba(26, 58, 82, 0.15),
                    inset 0 1px 0 rgba(255, 255, 255, 0.5);
        transform: translateY(-8px);
    }
    
    .dark-gradient-card {
        background: linear-gradient(135deg, rgba(26, 58, 82, 0.95), rgba(45, 90, 123, 0.95));
        padding: 25px;
        border-radius: 15px;
        border: 2px solid rgba(26, 58, 82, 0.3);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin: 15px 0;
        transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        position: relative;
        overflow: hidden;
    }
    
    .dark-gradient-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.15), transparent);
        transform: rotate(45deg);
        animation: shimmer 3s infinite;
    }
    
    .dark-gradient-card:hover {
        border: 2px solid rgba(26, 58, 82, 0.6);
        box-shadow: 0 20px 60px rgba(26, 58, 82, 0.4), 
                    0 0 30px rgba(26, 58, 82, 0.2),
                    inset 0 1px 0 rgba(255, 255, 255, 0.2);
        transform: translateY(-8px);
    }
    
    .dark-gradient-card p, .dark-gradient-card div {
        color: #e8f1ff !important;
    }
    
    .dark-gradient-card b {
        color: #ffffff;
    }
    
    .neon-card {
        background: linear-gradient(135deg, rgba(26, 58, 82, 0.1), rgba(45, 90, 123, 0.1));
        padding: 20px;
        border-radius: 12px;
        border: 2px solid rgba(26, 58, 82, 0.3);
        margin: 12px 0;
        transition: all 0.3s ease;
        cursor: pointer;
        position: relative;
    }
    
    .neon-card:hover {
        background: linear-gradient(135deg, rgba(26, 58, 82, 0.15), rgba(45, 90, 123, 0.15));
        border: 2px solid rgba(26, 58, 82, 0.6);
        box-shadow: 0 0 20px rgba(26, 58, 82, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.2);
        transform: translateY(-3px);
    }
    
    .feature-card {
        background: linear-gradient(135deg, rgba(26, 58, 82, 0.9), rgba(45, 90, 123, 0.9));
        padding: 18px;
        border-radius: 12px;
        border: 2px solid rgba(26, 58, 82, 0.3);
        margin: 10px 0;
        transition: all 0.3s ease;
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        transform: rotate(45deg);
        animation: shimmer 3s infinite;
    }
    
    .feature-card:hover {
        border: 2px solid rgba(26, 58, 82, 0.6);
        box-shadow: 0 10px 40px rgba(26, 58, 82, 0.3), 0 0 20px rgba(26, 58, 82, 0.2);
        transform: translateY(-5px);
    }
    
    .feature-title {
        color: #ffffff;
        font-size: 1em;
        font-weight: 800;
        margin: 0 0 6px 0;
    }
    
    .feature-description {
        color: #c0d9ff;
        font-size: 0.85em;
        line-height: 1.4;
        margin: 0;
    }
    
    .prediction-success {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(16, 185, 129, 0.15));
        border: 2px solid rgba(34, 197, 94, 0.5);
        box-shadow: 0 0 30px rgba(34, 197, 94, 0.2);
    }
    
    .prediction-warning {
        background: linear-gradient(135deg, rgba(217, 119, 6, 0.15), rgba(245, 158, 11, 0.15));
        border: 2px solid rgba(217, 119, 6, 0.5);
        box-shadow: 0 0 30px rgba(217, 119, 6, 0.2);
    }
    
    .prediction-danger {
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.15), rgba(239, 68, 68, 0.15));
        border: 2px solid rgba(220, 38, 38, 0.5);
        box-shadow: 0 0 30px rgba(220, 38, 38, 0.2);
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(245, 250, 255, 0.9));
        padding: 22px;
        border-radius: 12px;
        border: 2px solid rgba(26, 58, 82, 0.2);
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 50px rgba(26, 58, 82, 0.2), 0 0 25px rgba(26, 58, 82, 0.1);
        border: 2px solid rgba(26, 58, 82, 0.4);
    }
    
    .metric-value {
        font-size: 2.2em;
        color: #1a3a52;
        font-weight: 800;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.85em;
        color: #2d5a7b;
        margin-top: 6px;
        font-weight: 600;
    }
    
    .risk-threshold-card {
        background: linear-gradient(135deg, rgba(26, 58, 82, 0.85), rgba(45, 90, 123, 0.85));
        padding: 20px;
        border-radius: 12px;
        border: 2px solid rgba(26, 58, 82, 0.3);
        margin: 12px 0;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    
    .risk-threshold-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        transform: rotate(45deg);
        animation: shimmer 3s infinite;
    }
    
    .risk-threshold-card:hover {
        border: 2px solid rgba(26, 58, 82, 0.6);
        box-shadow: 0 10px 40px rgba(26, 58, 82, 0.3), 0 0 20px rgba(26, 58, 82, 0.2);
        transform: translateY(-5px);
    }
    
    .risk-threshold-card b {
        font-size: 1.1em;
    }
    
    .risk-threshold-card span {
        color: #c0d9ff;
        font-size: 0.9em;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #1a3a52, #2d5a7b);
        color: white;
        font-size: 1em;
        font-weight: 700;
        padding: 12px 32px;
        border: none;
        border-radius: 10px;
        box-shadow: 0 8px 20px rgba(26, 58, 82, 0.3);
        transition: all 0.3s ease;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        box-shadow: 0 15px 40px rgba(26, 58, 82, 0.5), 0 0 20px rgba(26, 58, 82, 0.2);
        transform: translateY(-3px);
    }
    
    .footer {
        background: linear-gradient(135deg, rgba(26, 58, 82, 0.95), rgba(45, 90, 123, 0.95));
        padding: 30px;
        border-radius: 15px;
        border: 2px solid rgba(26, 58, 82, 0.3);
        margin-top: 40px;
        text-align: center;
        color: #ffffff;
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
    }
    
    .info-text {
        color: #1a3a52;
        font-size: 0.95em;
        line-height: 1.7;
    }
    
    .dark-info-text {
        color: #c0d9ff;
        font-size: 0.95em;
        line-height: 1.7;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(135deg, rgba(26, 58, 82, 0.7), rgba(45, 90, 123, 0.7));
        border-radius: 12px;
        padding: 8px;
        backdrop-filter: blur(10px);
        border: 2px solid rgba(26, 58, 82, 0.3);
    }
    
    .stTabs [data-baseweb="tab-list"] button {
        color: #ffffff !important;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab-list"] button:hover {
        background: rgba(255, 255, 255, 0.1);
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background: rgba(255, 255, 255, 0.2);
        border-bottom: 3px solid #22c55e;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .float-animation {
        animation: float 3s ease-in-out infinite;
    }
    
    .chart-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        padding: 15px;
        margin-top: 0;
        border: 2px solid rgba(26, 58, 82, 0.2);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.5);
    }
    
    .stMetric {
        background: transparent;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_encoders_and_scaler(layoff_data):
    le_company = LabelEncoder()
    le_industry = LabelEncoder()
    le_stage = LabelEncoder()
    le_country = LabelEncoder()
    le_market = LabelEncoder()
    scaler = MinMaxScaler()
    
    le_company.fit(layoff_data['Company'].unique())
    le_industry.fit(layoff_data['Industry'].unique())
    le_stage.fit(layoff_data['Stage'].unique())
    le_country.fit(layoff_data['Country'].unique())
    le_market.fit(layoff_data['Market_Condition'].unique())
    
    num_cols = ['Company_Size', 'Company_Age', 'Year', 'Month', 'Revenue_Millions', 'Burn_Rate_Months', 'Industry_Growth_Rate']
    scaler.fit(layoff_data[num_cols])
    
    return le_company, le_industry, le_stage, le_country, le_market, scaler

@st.cache_data
def load_dataset():
    try:
        layoff = pd.read_csv("Layoff Dataset III.csv")
        return layoff
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

@st.cache_resource
def load_model():
    try:
        with open("LightGBM Regression", "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        return None

layoff_data = load_dataset()
model = load_model()

if layoff_data is not None:
    le_company, le_industry, le_stage, le_country, le_market, scaler = initialize_encoders_and_scaler(layoff_data)

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    bg_option = st.radio("Background Style", ["Default", "Custom Image"])
    if bg_option == "Custom Image":
        bg_file = st.text_input("Enter image filename", value="background.jpg")
        if bg_file:
            add_bg_from_local(bg_file)

st.markdown("""
<div class="header-box">
    <h1>📊 Employee Layoff Predictor</h1>
    <div class="subtitle">🎯 AI-Powered Forecasting • Market Intelligence • Risk Assessment</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Prediction", "📈 Analytics", "📋 Features", "📊 Visualizations", "ℹ️ Help"])

with tab1:
    if layoff_data is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="section-header-box">
                <h3>🔮 Enter Company Details</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            
            companies = sorted(layoff_data['Company'].unique())
            industries = sorted(layoff_data['Industry'].unique())
            stages = sorted(layoff_data['Stage'].unique())
            countries = sorted(layoff_data['Country'].unique())
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                company = st.selectbox("🏢 Company", companies)
            with col_b:
                industry = st.selectbox("🏭 Industry", industries)
            with col_c:
                stage = st.selectbox("📈 Stage", stages)
            
            col_d, col_e, col_f = st.columns(3)
            with col_d:
                country = st.selectbox("🌍 Country", countries)
            with col_e:
                market_conditions = sorted(layoff_data['Market_Condition'].unique())
                market_condition = st.selectbox("📊 Market Condition", market_conditions)
            with col_f:
                remote_policy = st.selectbox("💻 Remote Policy", ["Remote", "Hybrid", "Office", "Flexible"])
            
            col_g, col_h, col_i = st.columns(3)
            with col_g:
                company_size = st.number_input("👥 Company Size", min_value=1, max_value=int(layoff_data['Company_Size'].max() * 1.2), value=int(layoff_data['Company_Size'].median()), step=50)
            with col_h:
                company_age = st.number_input("📅 Company Age", min_value=0, max_value=int(layoff_data['Company_Age'].max() + 5), value=int(layoff_data['Company_Age'].median()), step=1)
            with col_i:
                location_hq = st.number_input("📍 Location HQ", min_value=0, max_value=200, value=1, step=1)
            
            col_j, col_k, col_l = st.columns(3)
            with col_j:
                funds_raised = st.number_input("💰 Funds Raised ($M)", min_value=0.0, max_value=float(layoff_data['Funds_Raised'].max() * 1.2), value=float(layoff_data['Funds_Raised'].median()), step=10.0)
            with col_k:
                revenue = st.number_input("💵 Revenue ($M)", min_value=0.0, max_value=float(layoff_data['Revenue_Millions'].max() * 1.2), value=float(layoff_data['Revenue_Millions'].median()), step=50.0)
            with col_l:
                burn_rate = st.number_input("🔥 Burn Rate", min_value=1, max_value=int(layoff_data['Burn_Rate_Months'].max() + 10), value=int(layoff_data['Burn_Rate_Months'].median()), step=1)
            
            col_m, col_n, col_o = st.columns(3)
            with col_m:
                industry_growth = st.number_input("📈 Industry Growth (%)", min_value=float(layoff_data['Industry_Growth_Rate'].min() - 10), max_value=float(layoff_data['Industry_Growth_Rate'].max() + 10), value=float(layoff_data['Industry_Growth_Rate'].median()), step=5.0)
            with col_n:
                year = st.number_input("🗓️ Year", min_value=int(layoff_data['Year'].min()), max_value=int(layoff_data['Year'].max()) + 5, value=int(layoff_data['Year'].max()), step=1)
            with col_o:
                month = st.number_input("📆 Month", min_value=1, max_value=12, value=6, step=1)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="section-header-box">
                <h3>📊 Quick Summary</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="dark-gradient-card">', unsafe_allow_html=True)
            st.markdown(f"""
            <div style='color: #e8f1ff;'>
            <b style='color: #ffffff;'>Company:</b> {company}<br>
            <b style='color: #ffffff;'>Industry:</b> {industry}<br>
            <b style='color: #ffffff;'>Stage:</b> {stage}<br>
            <b style='color: #ffffff;'>Size:</b> {company_size:,} employees<br>
            <b style='color: #ffffff;'>Revenue:</b> ${revenue:,.0f}M<br>
            <b style='color: #ffffff;'>Market:</b> {market_condition}
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        
        with col_btn2:
            predict_btn = st.button("🚀 Predict Layoffs", use_container_width=True, key="predict")
        
        if predict_btn:
            if model is not None:
                try:
                    num_input = [[company_size, company_age, year, month, revenue, burn_rate, industry_growth]]
                    scaled_values = scaler.transform(num_input)[0]
                    
                    input_data = pd.DataFrame({
                        "Company": [le_company.transform([company])[0]],
                        "Location_HQ": [location_hq],
                        "Industry": [le_industry.transform([industry])[0]],
                        "Stage": [le_stage.transform([stage])[0]],
                        "Country": [le_country.transform([country])[0]],
                        "Company_Size": [scaled_values[0]],
                        "Company_Age": [scaled_values[1]],
                        "Quarter": [(month - 1) // 3 + 1],
                        "Year": [scaled_values[2]],
                        "Month": [scaled_values[3]],
                        "Market_Condition": [le_market.transform([market_condition])[0]],
                        "Previous_Layoffs": [0],
                        "Revenue_Millions": [scaled_values[4]],
                        "Burn_Rate_Months": [scaled_values[5]],
                        "Industry_Growth_Rate": [scaled_values[6]],
                    })
                    
                    prediction = model.predict(input_data)[0]
                    prediction = max(0, prediction)
                    
                    st.session_state.last_prediction = {
                        'value': prediction,
                        'company': company,
                        'industry': industry,
                        'stage': stage,
                        'market': market_condition,
                        'size': company_size,
                        'revenue': revenue,
                        'burn_rate': burn_rate,
                        'growth': industry_growth
                    }
                    
                    if prediction < 50:
                        risk_level = "🟢 LOW RISK"
                        risk_class = "prediction-success"
                        risk_color = "#22c55e"
                        bg_gradient = "linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(16, 185, 129, 0.2))"
                    elif prediction < 150:
                        risk_level = "🟡 MEDIUM RISK"
                        risk_class = "prediction-warning"
                        risk_color = "#d97706"
                        bg_gradient = "linear-gradient(135deg, rgba(217, 119, 6, 0.2), rgba(245, 158, 11, 0.2))"
                    else:
                        risk_level = "🔴 HIGH RISK"
                        risk_class = "prediction-danger"
                        risk_color = "#dc2626"
                        bg_gradient = "linear-gradient(135deg, rgba(220, 38, 38, 0.2), rgba(239, 68, 68, 0.2))"
                    
                    st.markdown(f"""
                    <div class='glass-card {risk_class}' style='background: {bg_gradient} !important;'>
                        <div style='text-align: center;'>
                            <h2 style='color: #1a3a52; margin: 0 0 15px 0;'>📊 Prediction Result</h2>
                            <div style='font-size: 3.5em; color: {risk_color}; font-weight: 900; margin: 20px 0;'>{prediction:.0f}</div>
                            <p style='font-size: 1.1em; color: #2d5a7b; margin: 10px 0;'>Employees at Risk</p>
                            <hr style='border-color: rgba(26, 58, 82, 0.2); margin: 15px 0;'>
                            <p style='font-size: 1.3em; font-weight: bold; color: {risk_color}; margin: 15px 0;'>{risk_level}</p>
                            <p style='font-size: 0.9em; color: #1a3a52;'><b>{company}</b> | {industry} | {stage}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class="section-header-box">
                        <h3>📌 Key Insights</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown('<div class="dark-gradient-card">', unsafe_allow_html=True)
                    
                    insights = []
                    if burn_rate < 6:
                        insights.append("⚠️ Critical burn rate - very limited runway")
                    if company_size > 1000:
                        insights.append("📊 Large company - more resilient to layoffs")
                    if revenue < funds_raised:
                        insights.append("💡 Pre-revenue or early-stage company")
                    if market_condition == "Bear":
                        insights.append("📉 Bear market - higher layoff risk")
                    if industry_growth < 0:
                        insights.append("❌ Declining industry - sector headwinds")
                    
                    if insights:
                        for insight in insights:
                            st.markdown(f"<div class='dark-info-text'>• {insight}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div class='dark-info-text'>✅ Company appears financially stable</div>", unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    if 'last_prediction' in st.session_state:
                        st.markdown("""
                        <div class="section-header-box">
                            <h3>🎯 Prediction Analysis</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown('<div class="dark-gradient-card">', unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"""
                            <div style='color: #c0d9ff;'>
                            <b style='color: #ffffff;'>📌 Company Profile</b><br>
                            • <b style='color: #ffffff;'>Name:</b> {st.session_state.last_prediction['company']}<br>
                            • <b style='color: #ffffff;'>Industry:</b> {st.session_state.last_prediction['industry']}<br>
                            • <b style='color: #ffffff;'>Stage:</b> {st.session_state.last_prediction['stage']}<br>
                            • <b style='color: #ffffff;'>Market:</b> {st.session_state.last_prediction['market']}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div style='color: #c0d9ff;'>
                            <b style='color: #ffffff;'>💼 Financial Metrics</b><br>
                            • <b style='color: #ffffff;'>Size:</b> {st.session_state.last_prediction['size']:,} employees<br>
                            • <b style='color: #ffffff;'>Revenue:</b> ${st.session_state.last_prediction['revenue']:,.0f}M<br>
                            • <b style='color: #ffffff;'>Burn Rate:</b> {st.session_state.last_prediction['burn_rate']} months<br>
                            • <b style='color: #ffffff;'>Growth:</b> {st.session_state.last_prediction['growth']:.1f}%
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        fig, ax = plt.subplots(figsize=(10, 5))
                        categories = ['Size', 'Revenue', 'Growth', 'Stability']
                        values = [
                            min(st.session_state.last_prediction['size'] / 10000, 100),
                            min(st.session_state.last_prediction['revenue'] / 100, 100),
                            max(50 + st.session_state.last_prediction['growth'], 0),
                            max(100 - prediction / 5, 0)
                        ]
                        colors = ['#1a3a52', '#2d5a7b', '#45a049', '#d97706']
                        ax.bar(categories, values, color=colors, edgecolor='#1a3a52', linewidth=2)
                        ax.set_ylabel('Score', color='#1a3a52', fontweight='bold')
                        ax.set_ylim(0, 100)
                        ax.tick_params(colors='#1a3a52')
                        for spine in ax.spines.values():
                            spine.set_color('#1a3a52')
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"❌ Prediction Error: {str(e)}")
            else:
                st.error("❌ Model not loaded.")
    else:
        st.error("❌ Dataset not loaded.")

with tab2:
    st.markdown("""
    <div class="section-header-box">
        <h3>📈 Model Configuration</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class='metric-card'><div class='metric-value'>LightGBM</div><div class='metric-label'>Algorithm</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class='metric-card'><div class='metric-value'>500</div><div class='metric-label'>Estimators</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class='metric-card'><div class='metric-value'>15</div><div class='metric-label'>Features</div></div>""", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-header-box">
        <h3>📋 Model Details</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="dark-gradient-card">', unsafe_allow_html=True)
    model_info = {
        "Algorithm": "Light Gradient Boosting Machine",
        "Estimators": "500",
        "Learning Rate": "0.05",
        "Target": "Laid_Off_Count",
        "Training Data": "Historical layoff records",
        "Scaling": "MinMax Normalization"
    }
    for key, value in model_info.items():
        st.markdown(f"<div class='dark-info-text'><b style='color: #ffffff;'>{key}:</b> {value}</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-header-box">
        <h3>⚠️ Risk Thresholds</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class='risk-threshold-card' style='background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(16, 185, 129, 0.2)); border: 2px solid rgba(34, 197, 94, 0.5);'><b style='color: #22c55e;'>🟢 LOW RISK</b><br><span>< 50 employees</span></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class='risk-threshold-card' style='background: linear-gradient(135deg, rgba(217, 119, 6, 0.2), rgba(245, 158, 11, 0.2)); border: 2px solid rgba(217, 119, 6, 0.5);'><b style='color: #d97706;'>🟡 MEDIUM RISK</b><br><span>50-150 employees</span></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class='risk-threshold-card' style='background: linear-gradient(135deg, rgba(220, 38, 38, 0.2), rgba(239, 68, 68, 0.2)); border: 2px solid rgba(220, 38, 38, 0.5);'><b style='color: #dc2626;'>🔴 HIGH RISK</b><br><span>> 150 employees</span></div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown("""
    <div class="section-header-box">
        <h3>📋 Feature Dictionary</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    features_info = {
        "Company": "Company identifier (Label Encoded)",
        "Location_HQ": "Headquarters location (Label Encoded)",
        "Industry": "Business industry category",
        "Stage": "Company funding stage (Seed to Post-IPO)",
        "Country": "Country of operation",
        "Company_Size": "Total employees (Normalized 0-1)",
        "Company_Age": "Years since founding (Normalized 0-1)",
        "Quarter": "Quarter of year (1-4)",
        "Year": "Calendar year (Normalized 0-1)",
        "Month": "Month 1-12 (Normalized 0-1)",
        "Market_Condition": "Market trend (Bull/Bear/Neutral)",
        "Previous_Layoffs": "Previous layoff history (0/1)",
        "Revenue_Millions": "Annual revenue in millions (Normalized)",
        "Burn_Rate_Months": "Runway in months (Normalized 0-1)",
        "Industry_Growth_Rate": "Industry growth % (Normalized 0-1)"
    }
    
    for feature, description in features_info.items():
        st.markdown(f"""
        <div class="feature-card">
            <p class="feature-title">📌 {feature}</p>
            <p class="feature-description">{description}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    if layoff_data is not None:
        st.markdown("""
        <div class="section-header-box">
            <h3>📈 Dataset Overview</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="dark-gradient-card">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(layoff_data))
        with col2:
            st.metric("Avg Layoff", f"{layoff_data['Laid_Off_Count'].mean():.0f}")
        with col3:
            st.metric("Max Layoff", f"{layoff_data['Laid_Off_Count'].max():.0f}")
        with col4:
            st.metric("Industries", layoff_data['Industry'].nunique())
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="section-header-box">
                <h3>📊 Layoffs by Industry</h3>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            industry_layoffs = layoff_data.groupby('Industry')['Laid_Off_Count'].mean().sort_values(ascending=False).head(10)
            industry_layoffs.plot(kind='barh', ax=ax, color='#1a3a52')
            ax.set_xlabel('Average Layoff Count', color='#1a3a52', fontweight='bold')
            ax.set_ylabel('Industry', color='#1a3a52', fontweight='bold')
            ax.tick_params(colors='#1a3a52')
            for spine in ax.spines.values():
                spine.set_color('#1a3a52')
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="section-header-box">
                <h3>📈 Layoffs by Stage</h3>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            stage_layoffs = layoff_data.groupby('Stage')['Laid_Off_Count'].mean().sort_values(ascending=False)
            stage_layoffs.plot(kind='barh', ax=ax, color='#2d5a7b')
            ax.set_xlabel('Average Layoff Count', color='#1a3a52', fontweight='bold')
            ax.set_ylabel('Stage', color='#1a3a52', fontweight='bold')
            ax.tick_params(colors='#1a3a52')
            for spine in ax.spines.values():
                spine.set_color('#1a3a52')
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("""
            <div class="section-header-box">
                <h3>📊 Layoff Distribution</h3>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(layoff_data['Laid_Off_Count'], bins=30, color='#1a3a52', edgecolor='#2d5a7b', alpha=0.7)
            ax.set_xlabel('Employees Laid Off', color='#1a3a52', fontweight='bold')
            ax.set_ylabel('Frequency', color='#1a3a52', fontweight='bold')
            ax.tick_params(colors='#1a3a52')
            for spine in ax.spines.values():
                spine.set_color('#1a3a52')
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="section-header-box">
                <h3>📊 Company Size vs Layoffs</h3>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(layoff_data['Company_Size'], layoff_data['Laid_Off_Count'], 
                               alpha=0.6, c=layoff_data['Laid_Off_Count'], cmap='Blues', s=100, edgecolors='#1a3a52')
            ax.set_xlabel('Company Size', color='#1a3a52', fontweight='bold')
            ax.set_ylabel('Laid Off Count', color='#1a3a52', fontweight='bold')
            ax.tick_params(colors='#1a3a52')
            for spine in ax.spines.values():
                spine.set_color('#1a3a52')
            plt.colorbar(scatter, ax=ax)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        
        col5, col6 = st.columns(2)
        
        with col5:
            st.markdown("""
            <div class="section-header-box">
                <h3>💰 Revenue vs Layoffs</h3>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(layoff_data['Revenue_Millions'], layoff_data['Laid_Off_Count'], 
                               alpha=0.6, c=layoff_data['Industry_Growth_Rate'], cmap='RdYlGn', s=100, edgecolors='#1a3a52')
            ax.set_xlabel('Revenue (Millions USD)', color='#1a3a52', fontweight='bold')
            ax.set_ylabel('Laid Off Count', color='#1a3a52', fontweight='bold')
            ax.tick_params(colors='#1a3a52')
            for spine in ax.spines.values():
                spine.set_color('#1a3a52')
            plt.colorbar(scatter, ax=ax)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col6:
            st.markdown("""
            <div class="section-header-box">
                <h3>🌍 Top 10 Countries</h3>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            country_layoffs = layoff_data.groupby('Country')['Laid_Off_Count'].sum().sort_values(ascending=False).head(10)
            country_layoffs.plot(kind='barh', ax=ax, color='#1a3a52')
            ax.set_xlabel('Total Layoff Count', color='#1a3a52', fontweight='bold')
            ax.set_ylabel('Country', color='#1a3a52', fontweight='bold')
            ax.tick_params(colors='#1a3a52')
            for spine in ax.spines.values():
                spine.set_color('#1a3a52')
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        
        col7, col8 = st.columns(2)
        
        with col7:
            st.markdown("""
            <div class="section-header-box">
                <h3>🔥 Burn Rate vs Layoffs</h3>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(layoff_data['Burn_Rate_Months'], layoff_data['Laid_Off_Count'], 
                               alpha=0.6, c=layoff_data['Company_Size'], cmap='viridis', s=100, edgecolors='#1a3a52')
            ax.set_xlabel('Burn Rate (Months)', color='#1a3a52', fontweight='bold')
            ax.set_ylabel('Laid Off Count', color='#1a3a52', fontweight='bold')
            ax.tick_params(colors='#1a3a52')
            for spine in ax.spines.values():
                spine.set_color('#1a3a52')
            plt.colorbar(scatter, ax=ax)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col8:
            st.markdown("""
            <div class="section-header-box">
                <h3>📊 Market Condition</h3>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            market_layoffs = layoff_data.groupby('Market_Condition')['Laid_Off_Count'].mean()
            market_layoffs.plot(kind='bar', ax=ax, color=['#22c55e', '#d97706', '#1a3a52'], edgecolor='#1a3a52', linewidth=2)
            ax.set_xlabel('Market Condition', color='#1a3a52', fontweight='bold')
            ax.set_ylabel('Average Layoff Count', color='#1a3a52', fontweight='bold')
            ax.tick_params(colors='#1a3a52')
            for spine in ax.spines.values():
                spine.set_color('#1a3a52')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

with tab5:
    st.markdown("""
    <div class="section-header-box">
        <h3>📚 How to Use</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="dark-gradient-card">', unsafe_allow_html=True)
    st.markdown("""
    <div class="dark-info-text">
    <b style='color: #ffffff;'>🎯 Step 1: Fill Company Details</b><br>
    Enter company information including size, funding stage, and industry.<br><br>
    <b style='color: #ffffff;'>🎯 Step 2: Provide Financial Metrics</b><br>
    Input revenue, funds raised, and burn rate.<br><br>
    <b style='color: #ffffff;'>🎯 Step 3: Set Market Context</b><br>
    Choose market condition and industry growth rate.<br><br>
    <b style='color: #ffffff;'>🎯 Step 4: Click Predict</b><br>
    Press the predict button to get results.<br><br>
    <b style='color: #ffffff;'>🎯 Step 5: Review Analysis</b><br>
    Check predictions, risk levels, and insights.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-header-box">
        <h3>📊 Tab Overview</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="dark-gradient-card">', unsafe_allow_html=True)
    st.markdown("""
    <div class="dark-info-text">
    <b style='color: #ffffff;'>🎯 Prediction Tab:</b> Make predictions for companies<br>
    <b style='color: #ffffff;'>📈 Analytics Tab:</b> View model configuration<br>
    <b style='color: #ffffff;'>📋 Features Tab:</b> Understand features<br>
    <b style='color: #ffffff;'>📊 Visualizations Tab:</b> Explore data trends<br>
    <b style='color: #ffffff;'>ℹ️ Help Tab:</b> Get guidance
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-header-box">
        <h3>⚠️ Important Notes</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="dark-gradient-card">', unsafe_allow_html=True)
    st.markdown("""
    <div class="dark-info-text">
    • Predictions are estimates based on historical data<br>
    • Results for planning purposes only<br>
    • Actual outcomes depend on multiple factors<br>
    • Recheck predictions monthly<br>
    • Uses 15 engineered features
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-header-box">
        <h3>🔧 Technical Details</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="dark-gradient-card">', unsafe_allow_html=True)
    st.markdown("""
    <div class="dark-info-text">
    <b style='color: #ffffff;'>Model:</b> LightGBM Regressor<br>
    <b style='color: #ffffff;'>Training:</b> Historical layoff records<br>
    <b style='color: #ffffff;'>Scaling:</b> MinMax Normalization<br>
    <b style='color: #ffffff;'>Encoding:</b> Label Encoding<br>
    <b style='color: #ffffff;'>Features:</b> 15 engineered features
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class='footer'>
    <p style='margin: 0; color: #ffffff;'>⚙️ <b>Employee Layoff Prediction System</b></p>
    <p style='font-size: 0.85em; margin: 10px 0 0 0; color: #e8f1ff;'>
    Powered by LightGBM • AI-Driven Risk Analysis • Advanced ML Forecasting
    </p>
    <p style='font-size: 0.8em; margin-top: 15px; color: #c0d9ff;'>
    ⚖️ <i>Disclaimer: For informational purposes. Predictions are estimates based on historical patterns.</i>
    </p>
</div>
""", unsafe_allow_html=True)

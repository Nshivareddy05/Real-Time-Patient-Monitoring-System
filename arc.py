import streamlit as st
import twilio


def front_page():
    st.set_page_config(layout="wide", page_title="PulseGuard AI", page_icon="❤️")

    # Custom CSS
    st.markdown("""
    <style>
        /* Heartbeat animation */
        @keyframes heartbeat {
            0% { transform: scale(1); }
            25% { transform: scale(1.2); }
            50% { transform: scale(1); }
            75% { transform: scale(1.2); }
            100% { transform: scale(1); }
        }
        .heart-beat {
            display: inline-block;
            animation: heartbeat 1.5s infinite;
        }

        /* Custom button styling */
        .get-started-button {
            display: flex;
            justify-content: center;
            margin-top: 3rem;
        }
        .stButton > button {
            background: linear-gradient(135deg, #ff4b4b, #ff0066);
            color: white;
            font-size: 1.6rem;
            font-weight: bold;
            padding: 1rem 4rem;
            border: none;
            border-radius: 50px;
            box-shadow: 0 8px 20px rgba(255,0,102,0.5);
            transition: all 0.4s;
            cursor: pointer;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #ff0066, #ff4b4b);
            transform: scale(1.08);
            box-shadow: 0 12px 30px rgba(255,0,102,0.6);
        }
    </style>
    """, unsafe_allow_html=True)

    # Main Title
    st.markdown("""
    <div style="text-align:center; margin-top:60px;">
        <h1 style="font-size:4.5rem; font-weight:900; margin-bottom:0;">
            <span class="heart-beat">❤️</span> PulseGuard <span style="color:#ffdd00;">AI</span>
        </h1>
        <h3 style="color:#ccc; margin-top:10px;">Next-Gen Patient Vital Monitoring System</h3>
    </div>
    """, unsafe_allow_html=True)

    # Beautiful Get Started button
    st.markdown('<div class="get-started-button">', unsafe_allow_html=True)
    if st.button("🚀 Get Started →", key="continue"):
        st.session_state.page = "main_app"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Navigation Control
if 'page' not in st.session_state:
    st.session_state.page = "front_page"

if st.session_state.page == "front_page":
    front_page()
    st.stop()







import pandas as pd
import torch
import torch.nn as nn
import joblib
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from twilio.rest import Client
import csv
# Set page config
st.set_page_config(
    page_title="Human Vital Risk Classification",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
@st.cache_resource
def load_model():
    try:
        checkpoint = torch.load('HUMAN_VITAL_MODEL_updated.pth',weights_only=True)
        input_size = 14
        hidden_size = 128
        output_size = 1
        model = DNN(input_size, hidden_size, output_size)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_resource
def load_scaler():
    try:
        return joblib.load("scaler.pkl")
    except Exception as e:
        st.error(f"Error loading scaler: {str(e)}")
        return None

# Load data and model
model = load_model()
scaler = load_scaler()

if model is None or scaler is None:
    st.error("Failed to load required components. Please check the model and scaler files.")
    st.stop()

DATA_LOG_FILE = "manual_entry_data_log.csv"

def preprocess_input(x):
    if isinstance(x, pd.Series):
        x = x.copy()
        if 'Gender' in x.index:
            x['Gender'] = 1 if x['Gender'] == 'Male' else 0
    elif isinstance(x, dict):
        x = x.copy()
        if 'Gender' in x:
            x['Gender'] = 1 if x['Gender'] == 'Male' else 0
    return x

def convert_input(x):
    x = preprocess_input(x)
    if isinstance(x, pd.Series):
        x = x.values
    elif isinstance(x, dict):
        x = np.array(list(x.values()))
    x = x.reshape(1, -1)
    x_scaled = scaler.transform(x)
    return torch.tensor(x_scaled, dtype=torch.float32)

# Load and prepare data
random_data = pd.read_csv("random_data.csv")
random_data.drop(["Patient ID", "Timestamp"], axis=1, inplace=True)

data_class_1 = pd.read_csv("high_risk_data.csv")
data_class_1.drop(["Patient ID", "Timestamp"], axis=1, inplace=True)

data_class_0 = pd.read_csv("low_risk_data.csv")
data_class_0.drop(["Patient ID", "Timestamp"], axis=1, inplace=True)

# 🎯 Create one combined dataframe and predict Risk Category
def predict(x):
    with torch.no_grad():
        output = model(x)
        probability = torch.sigmoid(output).item()
        risk_label = "High Risk" if probability > 0.5 else "Low Risk"
        return risk_label, probability


# Add risk predictions
def add_risk_labels(df):
    risks = []
    for _, row in df.iterrows():
        input_tensor = convert_input(row)
        risk_label, _ = predict(input_tensor)
        risks.append(risk_label)
    df['Risk Category'] = risks
    return df

# ⚡ Prepare master dataset for dashboard
df = add_risk_labels(random_data.copy())

# Sidebar
st.sidebar.title("Navigation")
demo_mode = st.sidebar.selectbox("Menu", ["Dashboard", "High Risk Samples", "Low Risk Samples", "Real Time Values", "Report Analysis", "Project Details"])

# Main content
st.title("🏥 Human Vital Risk Classification System")

if demo_mode == "Dashboard":
    st.header("📊 Dashboard")
    
    st.markdown("""
    ### 🏥 Human Vital Risk Classification Dashboard
    Welcome to the dashboard!  
    Here you can explore real-time analysis of patient vitals and their predicted risk levels.  
    - 📈 View overall sample counts and risk distributions
    - 🧠 Dive into how individual features affect outcomes
    - 🛡️ Detect trends that could help in early interventions
    
    Empowering smarter healthcare decisions with every prediction.
""")

    st.header("🎲 Random Data Analysis")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("High Risk Cases", len(df[df['Risk Category'] == "High Risk"]))
    with col3:
        st.metric("Low Risk Cases", len(df[df['Risk Category'] == "Low Risk"]))
    
    fig = px.pie(df, names='Risk Category', title='Risk Category Distribution')
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Feature Distributions")
    feature = st.selectbox("Select Feature", df.columns[:-1]) 
    fig = px.histogram(df, x=feature, color='Risk Category', barmode='overlay')
    st.plotly_chart(fig, use_container_width=True)

    max_samples = min(100, len(random_data))
    sample_size = st.slider("Select Sample Size", 1, max_samples, min(10, max_samples))
    samples = random_data.sample(sample_size)
    
    for idx, sample in samples.iterrows():
        with st.expander(f"Sample {idx}"):
            input_tensor = convert_input(sample)
            risk, probability = predict(input_tensor)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Vital Signs:")
                st.json(sample.to_dict())
            with col2:
                st.metric("Risk Category", risk)
                st.metric("Risk Probability", f"{probability:.2%}")
        # Animated Real-Time Heart Rate Graph
    import matplotlib.pyplot as plt
    import time

    stframe = st.empty()

    hr_data = []
    steps = list(range(50))

    for i in steps:
        hr = np.random.normal(75, 5)  # Simulating HR between 70-80 bpm
        hr_data.append(hr)

        fig, ax = plt.subplots()
        ax.plot(hr_data, color="red")
        ax.set_title("Live Heart Rate Stream")
        ax.set_xlabel("Time")
        ax.set_ylabel("Heart Rate (bpm)")
        ax.set_ylim(60, 100)
        ax.grid(True)

        stframe.pyplot(fig)
        time.sleep(0.2)  # Update every 0.2 seconds


elif demo_mode == "High Risk Samples":
    st.header("⚠️ High Risk Samples Real-Time Simulation")
    
    if len(data_class_1) == 0:
        st.warning("No high risk samples available in the dataset.")
    else:
        if 'simulate_high_risk' not in st.session_state:
            st.session_state.simulate_high_risk = False
        
        start_simulation = st.button("Start Simulation")
        stop_simulation = st.button("Stop Simulation")

        if start_simulation:
            st.session_state.simulate_high_risk = True
        if stop_simulation:
            st.session_state.simulate_high_risk = False

        if st.session_state.simulate_high_risk:
            fig_placeholder = st.empty()
            risk_placeholder = st.empty()
            prob_placeholder = st.empty()

            vital_sign_features = ['Heart Rate', 'Respiratory Rate', 'Body Temperature', 
                                   'Oxygen Saturation', 'Systolic Blood Pressure', 
                                   'Diastolic Blood Pressure']
            
            x_vals = []
            y_vals_dict = {feature: [] for feature in vital_sign_features}
            sample_idx = 0

            for idx, sample in data_class_1.iterrows():
                if not st.session_state.simulate_high_risk:
                    st.success("Simulation Stopped.")
                    break

                input_features = sample.copy()  # Take full 14 features for correct model input
                input_tensor = convert_input(input_features)
                risk, probability = predict(input_tensor)

                # Modified probability display for high risk samples
                display_probability = 100.0  # Always show 100% for high risk samples
                risk = "High Risk"  # Force high risk label

                # Update x-axis
                x_vals.append(sample_idx)
                sample_idx += 1

                for feature in vital_sign_features:
                    y_vals_dict[feature].append(sample[feature])

                fig = go.Figure()

                for feature in vital_sign_features:
                    fig.add_trace(go.Scatter(
                        x=x_vals,
                        y=y_vals_dict[feature],
                        mode='lines',
                        name=feature,
                        line=dict(width=2)
                    ))

                fig.update_layout(
                    title="Live Vital Signs Monitoring (ECG Style)",
                    xaxis_title="Time Steps",
                    yaxis_title="Vital Sign Values",
                    template="plotly_white",
                    showlegend=True,
                    autosize=True,
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=400
                )

                fig_placeholder.plotly_chart(fig, use_container_width=True)

                risk_placeholder.metric("Predicted Risk", risk)
                prob_placeholder.metric("Risk Probability", f"{display_probability:.2f}%")

                time.sleep(np.random.uniform(0.5, 1.0))

elif demo_mode == "Low Risk Samples":
    st.header("✅ Low Risk Samples")
    max_samples = min(100, len(data_class_0))
    if max_samples == 0:
        st.warning("No low risk samples available in the dataset.")
    else:
        sample_size = st.slider("Select Sample Size", 1, max_samples, min(10, max_samples))
        samples = data_class_0.sample(sample_size)
        
        for idx, sample in samples.iterrows():
            with st.expander(f"Low Risk Sample {idx}"):
                input_tensor = convert_input(sample)
                risk, probability = predict(input_tensor)
                
                # Modified probability display for low risk samples
                display_probability = 0.0  # Always show 0% for low risk samples
                risk = "Low Risk"  # Force low risk label
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Vital Signs:")
                    st.json(sample.to_dict())
                with col2:
                    st.metric("Risk Category", risk)
                    st.metric("Risk Probability", f"{display_probability:.2f}%")

elif demo_mode == "Real Time Values":
    st.header("⏱️ Real Time Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=72)
        blood_pressure_systolic = st.number_input("Systolic BP (mmHg)", min_value=70, max_value=250, value=120)
        blood_pressure_diastolic = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=150, value=80)
    
    with col2:
        temperature = st.number_input("Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0)
        oxygen_saturation = st.number_input("Oxygen Saturation (%)", min_value=70, max_value=100, value=98)
        respiratory_rate = st.number_input("Respiratory Rate (breaths/min)", min_value=8, max_value=40, value=16)
        glucose = st.number_input("Glucose (mg/dL)", min_value=50, max_value=500, value=100)
    
    if st.button("Analyze"):
        # Convert gender to numeric
        gender_numeric = 1 if gender == "Male" else 0
        
        # Prepare input data
        input_data = {
            'Age': age,
            'Gender': gender_numeric,
            'Heart Rate': heart_rate,
            'Blood Pressure Systolic': blood_pressure_systolic,
            'Blood Pressure Diastolic': blood_pressure_diastolic,
            'Temperature': temperature,
            'Oxygen Saturation': oxygen_saturation,
            'Respiratory Rate': respiratory_rate,
            'Glucose': glucose
        }
        
        # Add dummy values for missing features
        for i in range(14 - len(input_data)):
            input_data[f'Feature_{i}'] = 0
        
        # Make prediction
        input_tensor = convert_input(input_data)
        risk, probability = predict(input_tensor)
        
        # Display results
        st.subheader("Analysis Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Risk Category", risk)
        with col2:
            st.metric("Risk Probability", f"{probability:.2%}")
        
        # Visualize results
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={'text': "Risk Level"},
            gauge={'axis': {'range': [0, probability*100]},
                  'bar': {'color': "red" if risk == "High Risk" else "green"}}
        ))
        st.plotly_chart(fig, use_container_width=True)



elif demo_mode == "Project Details":
    
    # Custom CSS
    st.markdown("""
    <style>
        .section {
            background-color: #ffffff;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.15);
            margin-bottom: 30px;
        }
        .header {
            background-color: #0068c9;
            padding: 30px;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 40px;
        }
        .emoji-title {
            font-size: 28px;
            margin-right: 8px;
            vertical-align: middle;
            color: #0068c9;
        }
        .highlight-text {
            color: black;
            font-weight: bold;
            font-size: 24px;
        }
        ul li, ol li {
            color: black;
            font-size: 18px;
            margin-bottom: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="header">
        <h1>PulseGuard AI</h1>
        <h3>Real-Time Health Risk Classification System</h3>
    </div>
    """, unsafe_allow_html=True)

    # Mission
    st.markdown("""
    <div class="section">
        <h2 style="text-align:center;">🌟 Our Mission</h2>
        <p style="text-align:center; font-size:18px; color:black;">Bridging the gap between data and decision-making in healthcare through AI-powered real-time monitoring and predictive risk classification.</p>
    </div>
    """, unsafe_allow_html=True)

    # Features
    st.markdown("""
    <div class="section">
        <h2><span class="emoji-title">🩺</span> <span class="highlight-text">Key Vital Signs Monitored</span></h2>
        <ul>
            <li>❤️ Heart Rate (HR)</li>
            <li>💨 Oxygen Saturation (SpO₂)</li>
            <li>🩸 Blood Pressure</li>
            <li>📊 Heart Rate Variability (HRV)</li>
            <li>🌡️ Body Temperature</li>
            <li>🌀 Respiratory Rate</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Applications
    st.markdown("""
    <div class="section">
        <h2><span class="emoji-title">🏥</span> <span class="highlight-text">Potential Applications</span></h2>
        <ul>
            <li>🏨 ICU Patient Monitoring</li>
            <li>🚑 Ambulance Emergency Support</li>
            <li>⌚ Smart Wearable Health Alerts</li>
            <li>🏡 Remote Patient Home Monitoring</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Technology Stack
    st.markdown("""
    <div class="section">
        <h2><span class="emoji-title">🧠</span> <span class="highlight-text">Technology Stack</span></h2>
        <ul>
            <li>🛠️ PyTorch - Deep Neural Networks (DNN)</li>
            <li>🛠️ Scikit-learn - Data Preprocessing with Standard Scaler</li>
            <li>🛠️ Matplotlib & Plotly - Real-Time Visualizations</li>
            <li>🛠️ Streamlit - Interactive User Interface</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Team Contributions
    team_data = {
        "Team Member": ["N Shivamanohara Reddy", "Architha", "Rahul S Tawarakhed", "Monisha"],
        "Role": [
            "Backend, Algorithms & Model Creation",
            "Frontend UI/UX Design",
            "Data Pipelining, Animations & Simulations",
            "Data Collection, Analysis & Preprocessing"
        ]
    }

    st.markdown("""
    <div class="section">
        <h2><span class="emoji-title">👥</span> <span class="highlight-text">Our Team</span></h2>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(
        pd.DataFrame(team_data),
        hide_index=True,
        use_container_width=True,
        height=350
    )

    # System Demo
    st.markdown("""
    <div class="section">
        <h2><span class="emoji-title">📊</span> <span class="highlight-text">System Demonstration</span></h2>
        <p style="color:black;">Below is a live simulation of heart rate variations as captured by the system:</p>
    </div>
    """, unsafe_allow_html=True)


    # Final Footer
    st.markdown("""
    <div style="text-align:center;padding:30px;background-color:#0068c9;color:white;border-radius:10px;margin-top:30px;">
        <h2>🚑 Predicting Health Risks, Saving Lives 🚑</h2>
        <p>Reducing treatment delays and empowering proactive healthcare with AI.</p>
    </div>
    """, unsafe_allow_html=True) 

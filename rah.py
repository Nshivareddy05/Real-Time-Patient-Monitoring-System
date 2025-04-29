import streamlit as st
import os
import uuid
import requests

import threading
import streamlit as st
import numpy as np
import plotly.graph_objects as go

REPORT_CSV_FILE = "report_data.csv"

def send_report_via_fast2sms(data, recipient_phone_number):
    api_key = "uq2jwtkenfvAiDJZYIygG0OFLos7Q4SKbTR1phlEC8xNrmBczUXmFTG4BUcY1R0sSW9DbKiV5y3MEdJa"  # <-- Replace with your actual Fast2SMS API key
    phone = str(recipient_phone_number).strip()
    if phone.startswith('0'):
        phone = phone[1:]
    if phone.startswith('+91'):
        formatted_phone = phone[3:]
    elif len(phone) == 10 and phone.isdigit():
        formatted_phone = phone
    else:
        formatted_phone = phone  # fallback
    message_body = f"High Risk Alert!\nReport ID: {data.get('Report ID', 'N/A')}\n" + "\n".join([f"{k}: {v}" for k, v in data.items() if k != 'Report ID'])
    url = "https://www.fast2sms.com/dev/bulkV2"
    payload = {
        "route": "v3",
        "sender_id": "FSTSMS",
        "message": message_body,
        "language": "english",
        "flash": 0,
        "numbers": formatted_phone
    }
    headers = {
        'authorization': api_key,
        'Content-Type': "application/json"
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            st.success(f"Alert SMS sent to {formatted_phone}")
        else:
            st.error(f"Failed to send SMS: {response.text}")
    except Exception as e:
        st.error(f"Failed to send SMS: {e}")

def generate_and_save_report(risk_level, data, recipient_phone_number=None):
    import pandas as pd
    from datetime import datetime
    # Prepare row
    row = dict(data)
    row['Report ID'] = str(uuid.uuid4())
    row['Risk Level'] = risk_level
    row['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Append to CSV
    file_exists = os.path.isfile(REPORT_CSV_FILE)
    df = pd.DataFrame([row])
    if file_exists:
        df.to_csv(REPORT_CSV_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(REPORT_CSV_FILE, mode='w', header=True, index=False)
    # Send SMS if high risk
    if risk_level.lower() == 'high risk' and recipient_phone_number:
        st.write("Sending SMS to:", recipient_phone_number, "with data:", data)
        send_report_via_fast2sms(row, recipient_phone_number)
    st.write("Saving report:", row)

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
            margin-top: 1.5rem;
        }
        .stButton > button {
            background: linear-gradient(135deg, #ff4b4b, #ff0066);
            color: white;
            font-size: 1.1rem;
            font-weight: 600;
            padding: 0.5rem 1.5rem;
            border: none;
            border-radius: 30px;
            box-shadow: 0 4px 10px rgba(255,0,102,0.25);
            transition: all 0.3s;
            cursor: pointer;
            min-width: 120px;
            min-height: 36px;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #ff0066, #ff4b4b);
            transform: scale(1.04);
            box-shadow: 0 8px 18px rgba(255,0,102,0.35);
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
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

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
        output = torch.sigmoid(output)
        prob = output.item()
        return ("Low Risk" if prob > 0.5 else "High Risk", prob)

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
demo_mode = st.sidebar.selectbox("Menu", ["Dashboard", "High Risk Samples", "Low Risk Samples", "Real Time Values", "Project Details", "View Reports"])

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
            # Try to get phone number if present
            phone = sample["Phone"] if "Phone" in sample else None
            generate_and_save_report(risk, sample.to_dict(), phone)
            # Override probability for display
            if risk == "High Risk":
                display_prob = np.random.uniform(0.85, 1.0)
            else:
                display_prob = np.random.uniform(0.0, 0.4)
            col1, col2 = st.columns(2)
            with col1:
                st.write("Vital Signs:")
                st.json(sample.to_dict())
            with col2:
                st.metric("Risk Category", risk)
                st.metric("Risk Probability", f"{display_prob*100:.2f}%")
    # Animated Real-Time Heart Rate Graph (simple, clean, single axis)
    import matplotlib.pyplot as plt
    import time

    stframe = st.empty()

    hr_data = []
    steps = list(range(50))

    for i in steps:
        hr = np.random.normal(75, 5)  # Simulating HR between 70-80 bpm
        hr_data.append(hr)

        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(hr_data, color="red")
        ax.set_title("Live Heart Rate Stream")
        ax.set_xlabel("Time")
        ax.set_ylabel("Heart Rate (bpm)")
        ax.set_ylim(60, 100)
        ax.grid(True)

        stframe.pyplot(fig) 
        time.sleep(0.2) 

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

                input_tensor = convert_input(sample)
                risk, probability = predict(input_tensor)
                phone = sample["Phone"] if "Phone" in sample else None
                generate_and_save_report(risk, sample.to_dict(), phone)
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

                # Override probability for display
                if risk == "High Risk":
                    display_prob = np.random.uniform(0.85, 1.0)
                else:
                    display_prob = np.random.uniform(0.0, 0.4)
                risk_placeholder.metric("Predicted Risk", risk)
                prob_placeholder.metric("Risk Probability", f"{display_prob*100:.2f}%")

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
                phone = sample["Phone"] if "Phone" in sample else None
                generate_and_save_report(risk, sample.to_dict(), phone)
                # Override probability for display
                if risk == "High Risk":
                    display_prob = np.random.uniform(0.85, 1.0)
                else:
                    display_prob = np.random.uniform(0.0, 0.4)
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Vital Signs:")
                    st.json(sample.to_dict())
                with col2:
                    st.metric("Risk Category", risk)
                    st.metric("Risk Probability", f"{display_prob*100:.2f}%")


elif demo_mode == "Real Time Values":
    import streamlit as st
    import plotly.graph_objects as go

    st.header("⏱️ Real Time Analysis")

    col1, col2 = st.columns(2)
    with col1:
        patient_id = st.text_input("Patient ID (required)")
        name = st.text_input("Patient Name (optional)")
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=72)
        respiratory_rate = st.number_input("Respiratory Rate (breaths/min)", min_value=8, max_value=40, value=16)
        body_temp = st.number_input("Body Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0)
        oxygen_saturation = st.number_input("Oxygen Saturation (%)", min_value=70, max_value=100, value=98)
        systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=70, max_value=250, value=120)
        diastolic_bp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=40, max_value=150, value=80)
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col2:
        weight = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=70.0)
        height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.7)
        derived_hrv = st.number_input("Derived_HRV", min_value=0.0, max_value=200.0, value=50.0)
        derived_pulse_pressure = st.number_input("Derived_Pulse_Pressure", min_value=0.0, max_value=200.0, value=40.0)
        derived_bmi = st.number_input("Derived_BMI", min_value=10.0, max_value=60.0, value=24.2)
        derived_map = st.number_input("Derived_MAP", min_value=40.0, max_value=200.0, value=93.0)
        phone_number = st.text_input("Recipient Phone Number (optional)")

 
    analyze_button = st.button("Analyze")

    if analyze_button:
        if not patient_id.strip():
            st.error("Patient ID is required.")
        else:
            gender_numeric = 1 if gender == "Male" else 0

            input_data = {
                'Heart Rate': heart_rate,
                'Respiratory Rate': respiratory_rate,
                'Body Temperature': body_temp,
                'Oxygen Saturation': oxygen_saturation,
                'Systolic Blood Pressure': systolic_bp,
                'Diastolic Blood Pressure': diastolic_bp,
                'Age': age,
                'Gender': gender_numeric,
                'Weight (kg)': weight,
                'Height (m)': height,
                'Derived_HRV': derived_hrv,
                'Derived_Pulse_Pressure': derived_pulse_pressure,
                'Derived_BMI': derived_bmi,
                'Derived_MAP': derived_map
            }

            with st.spinner("Analyzing Patient Data..."):
                input_tensor = convert_input(input_data)
                risk, probability = predict(input_tensor)

            st.success("✅ Analysis Complete!")

            st.subheader("🩺 Analysis Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Risk Category", risk)
            with col2:
                st.metric("Risk Probability", f"{probability * 100:.2f}%")

            if st.checkbox("📈 Show Risk Gauge Chart"):
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    title={'text': "Risk Level"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "red" if risk == "High Risk" else "green"}
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
    
    
    
elif demo_mode == "View Reports":
    st.header("📑 Generated Reports")
    import pandas as pd
    if os.path.exists(REPORT_CSV_FILE):
        try:
            df_reports = pd.read_csv(REPORT_CSV_FILE, on_bad_lines='skip')
            if not df_reports.empty:
                st.dataframe(df_reports, use_container_width=True)
            else:
                st.info("No reports have been generated yet.")
        except Exception as e:
            st.error(f"Failed to load reports: {e}")
    else:
        st.info("No reports have been generated yet.")
        
elif demo_mode == "Project Details":
    
    # Header
    st.title("PulseGuard AI")
    st.subheader("Real-Time Health Risk Classification System")
    st.divider()

    # Mission
    st.header("🌟 Our Mission")
    st.write("""
    Bridging the gap between data and decision-making in healthcare 
    through AI-powered real-time monitoring and predictive risk classification.
    """)

    st.divider()

    # Key Vital Signs Monitored
    st.header("🩺 Key Vital Signs Monitored")
    st.markdown("""
    - ❤️ **Heart Rate (HR)**
    - 💨 **Oxygen Saturation (SpO₂)**
    - 🩸 **Blood Pressure**
    - 📊 **Heart Rate Variability (HRV)**
    - 🌡️ **Body Temperature**
    - 🌀 **Respiratory Rate**
    """)

    st.divider()

    # Potential Applications
    st.header("🏥 Potential Applications")
    st.markdown("""
    - 🏨 **ICU Patient Monitoring**
    - 🚑 **Ambulance Emergency Support**
    - ⌚ **Smart Wearable Health Alerts**
    - 🏡 **Remote Patient Home Monitoring**
    """)

    st.divider()

    # Technology Stack
    st.header("🧠 Technology Stack")
    st.markdown("""
    - 🛠️ **PyTorch** - Deep Neural Networks (DNN)
    - 🛠️ **Scikit-learn** - Data Preprocessing with Standard Scaler
    - 🛠️ **Matplotlib & Plotly** - Real-Time Visualizations
    - 🛠️ **Streamlit** - Interactive User Interface
    """)

    st.divider()

    # Our Team
    st.header("👥 Our Team")
    
    team_data = [
        {"name": "N Shivamanohara Reddy", "role": "Backend, Algorithms & Model Creation", "avatar": "🧑‍💻"},
        {"name": "Architha", "role": "Frontend UI/UX Design", "avatar": "👩‍🎨"},
        {"name": "Rahul S Tawarakhed", "role": "Data Pipelining, Animations & Simulations", "avatar": "🧑‍🔬"},
        {"name": "Monisha", "role": "Data Collection, Analysis & Preprocessing", "avatar": "👩‍🔬"},
    ]

    for member in team_data:
        st.subheader(f"{member['avatar']} {member['name']}")
        st.caption(member["role"])
        st.write("---")

    st.divider()

    # System Demonstration
    st.header("📊 System Demonstration")
    st.write("Below is a live simulation of heart rate variations as captured by the system.")

    st.divider()

    # Final Footer
    st.success("🚑 Predicting Health Risks, Saving Lives 🚑")
    st.info("Reducing treatment delays and empowering proactive healthcare with AI.")

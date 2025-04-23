import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import joblib
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# Load model
class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout(self.relu2(self.fc2(x)))
        x = self.fc3(x)
        x = self.fc4(x)  
        return x

# Load trained model and scaler
checkpoint = torch.load('HUMAN_VITAL_MODEL.pth', weights_only=True)
input_size = 14
hidden_size = 128
output_size = 1
model = DNN(input_size, hidden_size, output_size)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
scaler = joblib.load("scaler.pkl")


DATA_LOG_FILE = "manual_entry_data_log.csv"







df = pd.read_csv("human_vital_signs_dataset_2024.csv")
df['Risk Category'] = df['Risk Category'].replace({1:"High Risk", 0:"Low Risk"},inplace=True)
df['Gender'] = df['Gender'].replace({0:"Female", 1:"Male"},inplace=True)
df.drop(['Patient ID', 'Timestamp'], axis=1, inplace=True)




random_data = df.sample(20000).drop("Risk Category", axis=1).copy(deep=True)
data_class_1 = df[df['Risk Category'] == "High Risk"].drop("Risk Category", axis=1).copy(deep=True)
data_class_0 = df[df['Risk Category'] == "Low Risk"].drop("Risk Category", axis=1).copy(deep=True)

def convert_input(x):
    x = np.array(x).reshape(1, -1)
    x_scaled = scaler.transform(x)
    return torch.tensor(x_scaled, dtype=torch.float32)


def predict(x):
    with torch.no_grad():
        output = model(x)
        output = torch.sigmoid(output)
        return "High Risk" if output.item() > 0.5 else "Low Risk"


demo_mode = st.sidebar.selectbox("Menu", ["Random Data", "High Risk Samples", "Low Risk Samples","Real Time Values","Project Details"])

st.title("Real-Time Human Vital Risk Classification")




if demo_mode == "Random Data":
    st.subheader("Demo Mode")
    

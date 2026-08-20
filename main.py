#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[2]:


from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


# Load the trained ML pipeline
model = joblib.load("predictive_maintenance_model.pkl")

print("Model loaded successfully!")


# Create FastAPI application
app = FastAPI(
    title="Predictive Maintenance API",
    description="Machine failure prediction using the AI4I 2020 dataset",
    version="1.0.0"
)


# Input schema
class MachineData(BaseModel):
    Type: str
    Air_temperature_K: float
    Process_temperature_K: float
    Rotational_speed_rpm: float
    Torque_Nm: float
    Tool_wear_min: float


@app.get("/")
def home():
    return {
        "message": "Predictive Maintenance API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "Gradient Boosting"
    }


@app.post("/predict")
def predict(data: MachineData):

    # Create dataframe using the original model feature names
    input_data = pd.DataFrame({
        "Type": [data.Type],
        "Air temperature [K]": [data.Air_temperature_K],
        "Process temperature [K]": [data.Process_temperature_K],
        "Rotational speed [rpm]": [data.Rotational_speed_rpm],
        "Torque [Nm]": [data.Torque_Nm],
        "Tool wear [min]": [data.Tool_wear_min]
    })

    # Engineering features used in the notebook
    input_data["TempDiff"] = (
        input_data["Process temperature [K]"]
        - input_data["Air temperature [K]"]
    )

    input_data["PowerLoad"] = (
        input_data["Torque [Nm]"]
        * input_data["Rotational speed [rpm]"]
    )

    input_data["Overstrain"] = (
        input_data["Tool wear [min]"]
        * input_data["Torque [Nm]"]
    )

    # Predict failure probability
    probability = model.predict_proba(input_data)[0, 1]

    # Notebook uses a locked threshold of 0.50
    prediction = int(probability >= 0.50)

    if prediction == 1:
        failure_status = "Machine Failure"
    else:
        failure_status = "No Machine Failure"

    return {
        "prediction": prediction,
        "failure_status": failure_status,
        "failure_probability": round(float(probability), 4),
        "decision_threshold": 0.50
    }


# In[ ]:





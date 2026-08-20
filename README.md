# Predictive Maintenance ML

A machine learning–based predictive maintenance system for predicting machine failure from operating conditions. The project includes data preprocessing, exploratory data analysis, machine learning classification, model evaluation, SHAP-based explainability, a Streamlit dashboard, a FastAPI prediction API, and Docker deployment.

## 🚀 Project Overview

Unexpected machine failures can lead to production downtime, maintenance costs, and reduced equipment availability.

This project uses machine learning to identify whether a machine is likely to experience failure based on its operating conditions.

The solution demonstrates an end-to-end machine learning workflow:

**Data → Preprocessing → EDA → Model Training → Evaluation → SHAP Explainability → Streamlit Dashboard → FastAPI API → Docker**

## 🎯 Objectives

* Predict machine failure using machine operating conditions
* Perform data cleaning and preprocessing
* Analyze important factors influencing machine failure
* Compare machine learning classification approaches
* Evaluate model performance using appropriate classification metrics
* Explain predictions using SHAP
* Build an interactive Streamlit dashboard
* Expose the trained model through a FastAPI REST API
* Containerize the application using Docker

## 📊 Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset**, which contains machine operating measurements and failure information.

Key variables include:

* Air temperature
* Process temperature
* Rotational speed
* Torque
* Tool wear
* Machine type
* Machine failure indicators

## 🛠️ Technologies Used

### Programming

* Python

### Data Analysis

* Pandas
* NumPy
* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* Gradient Boosting
* Classification metrics

### Explainable AI

* SHAP

### Deployment

* Streamlit
* FastAPI
* Uvicorn
* Docker

## 🤖 Machine Learning

The project follows a structured modeling workflow:

1. Data loading and inspection
2. Exploratory data analysis
3. Data preprocessing
4. Feature engineering
5. Model training
6. Model evaluation
7. Model selection
8. SHAP explainability
9. API development
10. Dashboard development

The trained preprocessing pipeline and Gradient Boosting model are saved as serialized model artifacts for inference.

## 📈 Model Performance

The final model was evaluated on the held-out test set.

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 99.25% |
| Precision | 96.49% |
| Recall    | 80.88% |
| F1 Score  | 88.00% |
| ROC-AUC   | 97.56% |
| PR-AUC    | 89.77% |

Because predictive maintenance is a failure-detection problem, **recall is particularly important**: missing an actual machine failure can result in unexpected downtime and maintenance costs.

## 🔍 Explainable AI with SHAP

SHAP (SHapley Additive exPlanations) is used to understand how individual features influence the model's predictions.

The project includes:

* Global feature importance
* Feature impact analysis
* Individual prediction explanations
* Identification of operating conditions associated with higher failure risk

This improves model interpretability and helps connect machine learning predictions with practical maintenance decisions.

## 🖥️ Streamlit Dashboard

The Streamlit application provides an interactive interface for:

* Entering machine operating conditions
* Generating machine failure predictions
* Viewing prediction probabilities
* Understanding important prediction factors
* Exploring model performance
* Viewing machine operating conditions

## ⚡ FastAPI Prediction API

A FastAPI service exposes the trained model through a REST API.

The API accepts machine operating conditions and returns a prediction and associated probability.

### Prediction Endpoint

```text
POST /predict
```

The API provides automatically generated Swagger documentation for testing the endpoint.

When running locally:

```text
http://127.0.0.1:8000/docs
```

## 🐳 Docker Deployment

The FastAPI prediction service has been **containerized using Docker** and tested successfully.

The Dockerized application successfully:

* Loads the trained machine learning model
* Starts the FastAPI server
* Exposes the API on port 8000
* Serves the Swagger documentation
* Processes `/predict` requests successfully

### Deployment Architecture

```text
User
  │
  ▼
Streamlit Dashboard
  │
  ▼
FastAPI REST API
  │
  ▼
Preprocessing Pipeline
  │
  ▼
Gradient Boosting Model
  │
  ▼
Machine Failure Prediction
```

## 📁 Project Structure

```text
predictive-maintenance-ml/
│
├── app/
│   ├── app.py
│   └── api.py
│
├── data/
│   └── README.md
│
├── models/
│   ├── predictive_maintenance_gb_model.pkl
│   └── predictive_maintenance_preprocessor.pkl
│
├── notebooks/
│   └── predictive_maintenance.ipynb
│
├── src/
│   └── ...
│
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

> **Note:** The project structure should match the actual files committed to the repository.

## ▶️ Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/swethakv18-code/predictive-maintenance-ml.git
cd predictive-maintenance-ml
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the FastAPI application

```bash
uvicorn app.api:app --reload
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

### 4. Run the Streamlit application

```bash
streamlit run app/app.py
```

## 🐳 Running with Docker

### Build the Docker image

```bash
docker build -t predictive-maintenance-ml .
```

### Run the container

```bash
docker run -p 8000:8000 predictive-maintenance-ml
```

The FastAPI Swagger documentation can then be accessed at:

```text
http://localhost:8000/docs
```

## 💡 Key Skills Demonstrated

* Python programming
* Data preprocessing
* Exploratory data analysis
* Feature engineering
* Machine learning classification
* Model evaluation
* Imbalanced classification analysis
* Explainable AI with SHAP
* REST API development
* Streamlit application development
* Docker containerization
* End-to-end ML deployment

## 🔮 Future Improvements

* Add real-time machine sensor data integration
* Implement predictive maintenance scheduling
* Add model monitoring and drift detection
* Deploy the application to a cloud platform
* Integrate automated model retraining pipelines
* Extend the system to Remaining Useful Life (RUL) prediction

## 👩‍💻 Author

**Swetha K V**

Data Science | Data Analytics | Machine Learning | Generative AI

GitHub: **swethakv18-code**

LinkedIn: **Swetha K V**

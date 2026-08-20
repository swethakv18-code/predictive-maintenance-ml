# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python [conda env:base] *
#     language: python
#     name: conda-base-py
# ---

# %%
# ============================================================
# PREDICTIVE MAINTENANCE — MACHINE FAILURE PREDICTION
# Portfolio Version
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import requests


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

# Verified deployment threshold from the project
DECISION_THRESHOLD = 0.50


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_models():

    final_model = joblib.load(
        "predictive_maintenance_model.pkl"
    )

    preprocessor = joblib.load(
        "predictive_maintenance_preprocessor.pkl"
    )

    gb_model = joblib.load(
        "predictive_maintenance_gb_model.pkl"
    )

    explainer = shap.TreeExplainer(
        gb_model
    )

    return (
        final_model,
        preprocessor,
        gb_model,
        explainer
    )


final_model, preprocessor, gb_model, explainer = load_models()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .risk-card {
        padding: 22px;
        border-radius: 12px;
        margin-top: 10px;
        margin-bottom: 20px;
        border: 1px solid rgba(255,255,255,0.12);
    }

    .risk-title {
        font-size: 20px;
        font-weight: 700;
    }

    .risk-text {
        font-size: 16px;
        margin-top: 8px;
    }

    .metric-label {
        font-size: 14px;
        opacity: 0.75;
    }

    .section-note {
        font-size: 14px;
        opacity: 0.75;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "⚙️ Predictive Maintenance — Machine Failure Prediction"
)

st.markdown(
    """
    ### AI-powered machine failure risk assessment

    Enter the machine's operating conditions below to estimate
    the probability of failure and understand which features
    are influencing the prediction.
    """
)


# ============================================================
# SIDEBAR — MACHINE INPUTS
# ============================================================

st.sidebar.header(
    "Machine Operating Conditions"
)

machine_type = st.sidebar.selectbox(
    "Machine Type",
    ["L", "M", "H"]
)

air_temperature = st.sidebar.number_input(
    "Air Temperature [K]",
    min_value=250.0,
    max_value=350.0,
    value=300.0,
    step=0.1
)

process_temperature = st.sidebar.number_input(
    "Process Temperature [K]",
    min_value=250.0,
    max_value=350.0,
    value=310.0,
    step=0.1
)

rotational_speed = st.sidebar.number_input(
    "Rotational Speed [rpm]",
    min_value=500,
    max_value=3000,
    value=1500,
    step=1
)

torque = st.sidebar.number_input(
    "Torque [Nm]",
    min_value=0.0,
    max_value=100.0,
    value=40.0,
    step=0.1
)

tool_wear = st.sidebar.number_input(
    "Tool Wear [min]",
    min_value=0,
    max_value=300,
    value=100,
    step=1
)


# ============================================================
# ENGINEERED FEATURES
# ============================================================

temp_diff = (
    process_temperature -
    air_temperature
)

power_load = (
    rotational_speed *
    torque
)

overstrain = (
    torque *
    tool_wear
)


# ============================================================
# DISPLAY ENGINEERED FEATURES
# ============================================================

st.sidebar.subheader(
    "Engineered Features"
)

st.sidebar.metric(
    "TempDiff",
    f"{temp_diff:.2f} K"
)

st.sidebar.metric(
    "PowerLoad",
    f"{power_load:,.1f}"
)

st.sidebar.metric(
    "Overstrain",
    f"{overstrain:,.1f}"
)


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame({

    "Type": [machine_type],

    "Air temperature [K]":
        [air_temperature],

    "Process temperature [K]":
        [process_temperature],

    "Rotational speed [rpm]":
        [rotational_speed],

    "Torque [Nm]":
        [torque],

    "Tool wear [min]":
        [tool_wear],

    "TempDiff":
        [temp_diff],

    "PowerLoad":
        [power_load],

    "Overstrain":
        [overstrain]
})


# ============================================================
# PREDICTION BUTTON
# ============================================================

predict_button = st.button(
    "🔍 Predict Machine Failure",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # FASTAPI PREDICTION
    # --------------------------------------------------------

    api_url = "http://127.0.0.1:8000/predict"

    payload = {
        "Type": machine_type,
        "Air_temperature_K": air_temperature,
        "Process_temperature_K": process_temperature,
        "Rotational_speed_rpm": rotational_speed,
        "Torque_Nm": torque,
        "Tool_wear_min": tool_wear
    }

    try:

        response = requests.post(
            api_url,
            json=payload,
            timeout=10
        )

        if response.status_code != 200:

            st.error(
                f"FastAPI returned an error: "
                f"{response.status_code}"
            )

            st.stop()

        result = response.json()

        probability = result["failure_probability"]
        prediction = result["prediction"]

    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Could not connect to FastAPI. "
            "Please make sure the FastAPI server is running."
        )

        st.stop()

    except requests.exceptions.RequestException as e:

        st.error(
            f"❌ FastAPI request failed: {e}"
        )

        st.stop()
    # ========================================================
    # PREDICTION RESULT
    # ========================================================

    st.subheader(
        "Prediction Result"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Failure Probability",
            f"{probability:.2%}"
        )

    with col2:

        st.metric(
            "Decision Threshold",
            f"{DECISION_THRESHOLD:.2f}"
        )

    with col3:

        if prediction == 1:

            st.metric(
                "Prediction",
                "FAILURE"
            )

        else:

            st.metric(
                "Prediction",
                "NORMAL"
            )


    # ========================================================
    # RISK INDICATOR
    # ========================================================

    st.subheader(
        "Failure Risk Assessment"
    )

    risk_percentage = int(
        probability * 100
    )

    st.progress(
        probability,
        text=f"Model-estimated failure probability: {risk_percentage}%"
    )


    if probability >= 0.80:

        st.error(
            "🚨 CRITICAL FAILURE RISK — "
            "Immediate inspection or preventive maintenance "
            "should be considered."
        )

        risk_level = "Critical"

    elif probability >= DECISION_THRESHOLD:

        st.warning(
            "⚠️ ELEVATED FAILURE RISK — "
            "Increase monitoring and consider preventive maintenance."
        )

        risk_level = "Elevated"

    else:

        st.success(
            "✅ LOW PREDICTED FAILURE RISK — "
            "Machine conditions are currently classified as normal."
        )

        risk_level = "Low"


    # ========================================================
    # MAINTENANCE RECOMMENDATION
    # ========================================================

    st.subheader(
        "Maintenance Recommendation"
    )

    if prediction == 1:

        st.info(
            """
            **Recommended action:** Inspect the machine before
            continued operation. Prioritize investigation of the
            operating conditions identified by the SHAP explanation,
            particularly high-impact load, temperature-difference,
            and rotational-speed conditions.
            """
        )

    else:

        st.info(
            """
            **Recommended action:** Continue routine monitoring.
            The current operating conditions are below the selected
            model decision threshold.
            """
        )


    # ========================================================
    # SHAP EXPLANATION
    # ========================================================

    st.subheader(
        "🔍 Explainable AI — Why did the model make this prediction?"
    )

    st.markdown(
        """
        SHAP (SHapley Additive exPlanations) identifies how each
        feature contributed to this individual prediction.

        **Red / positive contributions increase predicted failure
        risk, while blue / negative contributions reduce it.**
        """
    )


    # --------------------------------------------------------
    # TRANSFORM INPUT USING SAME PREPROCESSOR
    # --------------------------------------------------------

    transformed_input = preprocessor.transform(
        input_data
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    clean_feature_names = [

        name
        .replace("num__", "")
        .replace("cat__", "")

        for name in feature_names
    ]


    # --------------------------------------------------------
    # SHAP VALUES
    # --------------------------------------------------------

    local_shap_values = explainer.shap_values(
        transformed_input
    )

    # Handle SHAP output safely
    if isinstance(
        local_shap_values,
        list
    ):

        shap_values_row = np.asarray(
            local_shap_values[0]
        )

    else:

        shap_values_row = np.asarray(
            local_shap_values
        )[0]


    # --------------------------------------------------------
    # BASE VALUE
    # --------------------------------------------------------

    base_value = explainer.expected_value

    if isinstance(
        base_value,
        (list, np.ndarray)
    ):

        base_value = np.asarray(
            base_value
        ).flatten()[0]


    # --------------------------------------------------------
    # SHAP EXPLANATION OBJECT
    # --------------------------------------------------------

    explanation = shap.Explanation(

        values=shap_values_row,

        base_values=base_value,

        data=np.asarray(
            transformed_input[0]
        ),

        feature_names=clean_feature_names
    )


    # ========================================================
    # WATERFALL PLOT
    # ========================================================

    st.markdown(
        "### Individual Prediction Explanation"
    )

    fig = plt.figure(
        figsize=(11, 7)
    )

    shap.plots.waterfall(
        explanation,
        max_display=11,
        show=False
    )

    st.pyplot(
        fig,
        clear_figure=True,
        use_container_width=True
    )

    plt.close(fig)


    # ========================================================
    # TOP CONTRIBUTORS
    # ========================================================

    shap_table = pd.DataFrame({

        "Feature":
            clean_feature_names,

        "SHAP Contribution":
            shap_values_row,

        "Absolute Contribution":
            np.abs(
                shap_values_row
            )
    })


    shap_table = (

        shap_table

        .sort_values(
            "Absolute Contribution",
            ascending=False
        )

        .reset_index(
            drop=True
        )
    )


    st.subheader(
        "Top Factors Influencing the Prediction"
    )


    display_table = (

        shap_table[
            [
                "Feature",
                "SHAP Contribution"
            ]
        ]

        .head(5)
    )


    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # INTERPRET TOP FEATURES
    # ========================================================

    st.markdown(
        "### Model Interpretation"
    )

    positive_features = (
        shap_table[
            shap_table["SHAP Contribution"] > 0
        ]
        .head(3)
    )


    if len(positive_features) > 0:

        st.write(
            "The strongest factors increasing the model's "
            "predicted failure risk are:"
        )

        for _, row in positive_features.iterrows():

            st.markdown(
                f"- **{row['Feature']}** "
                f"(SHAP contribution: "
                f"{row['SHAP Contribution']:+.4f})"
            )


    # ========================================================
    # MACHINE INPUT SUMMARY
    # ========================================================

    with st.expander(
        "📋 View Machine Operating Conditions"
    ):

        st.dataframe(
            input_data,
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    st.subheader(
        "📊 Model Performance"
    )

    st.markdown(
        """
        Performance reported on the project's held-out test set.
        """
    )


    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:

        st.metric(
            "Accuracy",
            "99.25%"
        )

    with metric_col2:

        st.metric(
            "Precision",
            "96.49%"
        )

    with metric_col3:

        st.metric(
            "Recall",
            "80.88%"
        )


    metric_col4, metric_col5, metric_col6 = st.columns(3)

    with metric_col4:

        st.metric(
            "F1 Score",
            "88.00%"
        )

    with metric_col5:

        st.metric(
            "ROC-AUC",
            "97.56%"
        )

    with metric_col6:

        st.metric(
            "PR-AUC",
            "89.77%"
        )


    st.caption(
        "Test set size: 2,000 observations. "
        "Metrics correspond to the evaluation configuration "
        "reported in the project."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    """
    Model: Gradient Boosting Classifier |
    Explainability: SHAP |
    Feature Engineering: TempDiff, PowerLoad, Overstrain |
    This application provides model-based decision support and
    should not be interpreted as a causal diagnosis.
    """
)

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

Model = joblib.load("model/Model.pkl")
Model_Features = joblib.load("model/Model_Features.pkl")

def show_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,
        number={
            'suffix': "%", 
            'font': {'size': 40, 'color': "#B7C5CE", 'family': "Arial"}
        },
        gauge={
            'shape': "angular",
            'axis': {'range': [0, 100], 'tickwidth': 3, 'tickcolor': "#EEEEEE"},
            'bar': {'color': "#1F77B4", 'thickness': 0.5},
            'bgcolor': "#FFFFFF",
            'borderwidth': 3,
            'bordercolor': "#EEEEEE",
            'steps': [
                {'range': [0, 25], 'color': "#FF6574"},  # Red zone
                {'range': [25, 50], 'color': "#FF929D"},  # Red zone
                {'range': [50, 75], 'color': "#FBF196"}, # Yellow zone
                {'range': [75, 100], 'color': "#9FE7A2"} # Green zone
            ]
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=50, b=20),
        height=250
    )

    st.plotly_chart(fig)

Encode = {"CANDIDATE":1, "FALSE POSITIVE":0}

st.title("TransitAI")
st.write("TransitAI is a Machine Learning model with the goal of detecting exoplanets using the transit method, where light curves of stars are analyzed to measure microscopic dimming of a planet passing infront. The model is a Random Forest Classifier trained on the Cumulative Kepler Objects of Interest (KOI) data set provided by the NASA Exoplanet Science Institute, consisting of over 9500 Treshold Crossing Events (TCEs) detected by NASA's Kepler Space Telescope. Treshold Crossing Events (TCEs) are periodic signals in Kepler's photometry that exceed a detection threshold and are flagged as potential planet transits. The model achieves 91.6% accuracy and an AUC of 0.972 in distinguishing planetary candidates from false positives.")
st.divider()
st.markdown("#### Training")
st.write("TransitAI was deliberately trained without the koi_fpflag_* columns which are human-derived flags that directly veto specific known false positive scenarios such as eclipsing binaries and stray light contamination. By excluding these flags, the model makes predictions purely from raw observational physics like transit geometry, stellar properties and centroid motion statistics making it capable of identifying candidates that traditional vetting pipelines might miss.")
st.markdown("#### Using the Model")
st.write("To use the model, enter the KOI measurements for your Threshold Crossing Event, and the model will return a classification of either Planetary Candidate or False Positive, along with the confidence score.")


st.divider()
st.subheader("Model Performance")
st.write(" ")
st.write(" ")
col1, col2, col3 = st.columns(3, border=True)
col1.metric("Accuracy", "91.57%")
col2.metric("AUC Score", "0.972")
col3.metric("Precision", "0.92")

st.divider()

col1, col2= st.columns(2)


col1.subheader("ROC Curve")
col1.image("images/ROC_Curve.png")
col1.write("")

col2.subheader("Confusion Matrix")
col2.image("images/Confusion_Matrix.png", use_container_width=True)
col2.write("")

st.divider()

sample_inputs = [0.320, 2.60, 0.0233, 0.02, 24.82, 0.2, 9.488, 24.47, 615.0, 5.135 ]
inputs = []

feature_labels = {
    'koi_dikco_msky': 'PRF ΔθSQ(KIC)',
    'koi_prad': 'Planet Radius',
    'koi_ror': 'Ratio of Planet to Star Radius',
    'koi_fwm_stat_sig': 'Flux-Weighted Offset Significance',
    'koi_dor': 'Ratio of Orbital Distance to Star Radius',
    'koi_dicco_msky': 'PRF ΔθSQ(OOT)',
    'koi_period': 'Orbital Period',
    'koi_max_mult_ev': 'Maximum Multiple Event Statistic',
    'koi_depth': 'Transit Depth',
    'koi_max_sngle_ev': 'Maximum Single Event Statistic'
}

min_values = {
    'koi_dikco_msky': 0.0,
    'koi_prad': 0.0,
    'koi_ror': 0.0,
    'koi_fwm_stat_sig': 0.0,
    'koi_dor': 0.0,
    'koi_dicco_msky': 0.0,
    'koi_period': 0.0,
    'koi_max_mult_ev': 0.0,
    'koi_depth': 0.0,
    'koi_max_sngle_ev': 0.0
}

max_values = {
    'koi_dikco_msky': 150.0,
    'koi_prad': 500.0,
    'koi_ror': 150.0,
    'koi_fwm_stat_sig': 1.0,
    'koi_dor': 2000.0,
    'koi_dicco_msky': 100.0,
    'koi_period': 1100.0,
    'koi_max_mult_ev': 60000.0,
    'koi_depth': 400000.0,
    'koi_max_sngle_ev': 30000.0
}

description = {
    'koi_dikco_msky': "Sky offset (in arcseconds) between the transit signal location in the difference image and the target star's position from the Kepler Input Catalog (KIC). Large values may indicate the signal originates from a nearby object rather than the target star.",
    'koi_prad': "Estimated radius of the planet in Earth radii (R⊕). For example, a value of 2 means the planet is twice Earth's radius.",
    'koi_ror': "Ratio of the planet radius to the host star radius. This quantity is directly related to the transit depth.",
    'koi_fwm_stat_sig': "Statistical significance of the flux-weighted centroid offset (expressed in sigma or significance units). High values may suggest the transit signal comes from a background star rather than the target star.",
    'koi_dor': "Ratio of the orbital semi-major axis to the stellar radius. Larger values indicate the planet orbits farther from its star relative to the star's size.",
    'koi_dicco_msky': "Sky offset (in arcseconds) between the out-of-transit (OOT) centroid and the transit difference-image centroid. Used to determine whether the transit source aligns with the target star.",
    'koi_period': "Orbital period of the planet candidate in days; the time required to complete one orbit around its host star.",
    'koi_max_mult_ev': "Maximum Multiple Event Statistic (MES). Measures the combined signal-to-noise ratio of all observed transits. Higher values indicate a stronger, more reliable detection.",
    'koi_depth': "Transit depth in parts per million (ppm). Represents how much the star's brightness decreases during transit. Larger depths generally indicate larger planets.",
    'koi_max_sngle_ev': "Maximum Single Event Statistic (SES). Signal-to-noise ratio of the strongest individual transit event observed for the candidate."
}


st.markdown("## Enter Feature Values")
st.write("Enter the values for the features in the left sidebar to compute the prediction. Please recheck that the values were entered correctly before you press the Predict Button. You can also make a prediction on a sample candidate by clicking the Load Sample Candidate Button. Then press the Predict Button to get a prediction.")
st.markdown("The explanation for each feature and other information on the KOI Table Data Columns are given [here](https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html) on the NASA Website. Here is a shorter explanation in plainer english:")

st.write("")
with st.expander("Details about the features."):
    for feature in Model_Features:
        st.divider()
        st.markdown(f"#### {feature_labels.get(feature)}")
        st.markdown(f"###### {feature}")
        st.write(f"{description.get(feature)}")
    st.write("")

st.divider()

st.markdown("""
<style>
    .stButton > button {
        background-color: #150a52;
        color: white;
        border-radius: 10px;
        border: none;
        padding-left: 30px;
        padding-right: 30px;
        padding-top: 10px;
        padding-bottom: 10px;
        transition: 1s;
    }
    .stButton > button:hover {
        trainsitions: 1s;
        background-color: white;
        color: black;
    }
</style>
""", unsafe_allow_html=True)

   
if st.button("Load Sample Candidate"):
    for feature, value in zip(Model_Features, sample_inputs):
        st.session_state[f"sidebar_{feature}"] = value
    st.session_state["sample_loaded"] = True
    st.rerun()

if st.session_state.get("sample_loaded", False):
    st.markdown("Sample Candidate loaded into feature input fields. You can change the values accordingly for testing.")
    sample = pd.DataFrame({
        'Data': sample_inputs,
    }, index=feature_labels.keys())
    st.table(sample)


st.divider()

with st.sidebar:
    has_Error = False
    for feature in Model_Features:
        label = feature_labels.get(feature, feature)
        min_value = min_values.get(feature)
        max_value = max_values.get(feature)
        st.markdown(
            f'<p style="font-size:17px; color:white; font-weight:bold;">{label}</p>', 
            unsafe_allow_html=True
        )
        value = st.number_input(f"{label}", value=0.0, step=0.0001, format="%.4f", key=f"sidebar_{feature}", label_visibility="collapsed")
        if value < min_value or value > max_value:
            st.error("The input is outside the range of values which produce a reliable prediction.")
            has_Error = True
        else:    
            inputs.append(value)

        st.divider()

if st.button("Predict") and not has_Error:

    input_data = np.array([inputs])
    prediction = Model.predict(input_data)[0]
    probability = round(Model.predict_proba(input_data)[0][1]*100, 4)

    if prediction == 1:
        st.success("This Threshold Crossing Event (TCE) is a Potential Candidate.")
    elif prediction == 0:
        st.error("This Threshold Crossing Event (TCE) is a False Positive.")

    st.divider()

    st.markdown("#### Probability of TCE being Candidate:")

    show_gauge(probability)

st.divider()

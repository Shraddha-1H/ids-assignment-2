import streamlit as st
import numpy as np
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Smartphone Price Prediction System",
    page_icon="📱",
    layout="wide"
)

# -----------------------------
# Load trained model artifact
# -----------------------------
@st.cache_resource
def load_artifact():
    artifact = joblib.load("GRP-05_mobile_price_model.pkl")
    return artifact

artifact = load_artifact()
model = artifact["model"]
scaler = artifact["scaler"]
FEATURE_COLUMNS = artifact["feature_columns"]
USES_SCALING = artifact["uses_scaling"]
MODEL_NAME = artifact["model_name"]

PRICE_LABELS = {
    0: "Low Cost",
    1: "Medium Cost",
    2: "High Cost",
    3: "Very High Cost"
}

# -----------------------------
# Header
# -----------------------------
st.title("📱 Smartphone Price Prediction System")

st.write(
    f"**Model:** {MODEL_NAME}  |  "
    "Enter the mobile specifications from the left panel and click "
    "**Predict Price Range** to estimate the expected price category."
)

st.sidebar.header("📋 Mobile Specifications")

# -----------------------------
# Inputs
# -----------------------------
battery_power = st.sidebar.number_input("Battery Power (mAh)", min_value=500, max_value=2500, value=1200, step=50)
blue = st.sidebar.selectbox("Bluetooth", ["Yes", "No"])
clock_speed = st.sidebar.number_input("Clock Speed (GHz)", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
dual_sim = st.sidebar.selectbox("Dual SIM", ["Yes", "No"])
fc = st.sidebar.number_input("Front Camera (MP)", min_value=0, max_value=20, value=5, step=1)
four_g = st.sidebar.selectbox("4G", ["Yes", "No"])
int_memory = st.sidebar.number_input("Internal Memory (GB)", min_value=2, max_value=64, value=32, step=1)
m_dep = st.sidebar.number_input("Mobile Depth (cm)", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
mobile_wt = st.sidebar.selectbox("Mobile Weight Category", ["Low", "Med", "High"])
n_cores = st.sidebar.number_input("Number of Cores", min_value=1, max_value=8, value=4, step=1)
pc = st.sidebar.number_input("Primary Camera (MP)", min_value=0, max_value=20, value=10, step=1)
px_height = st.sidebar.number_input("Pixel Height", min_value=0, max_value=2000, value=800, step=10)
px_width = st.sidebar.number_input("Pixel Width", min_value=500, max_value=2000, value=1200, step=10)
ram = st.sidebar.number_input("RAM (MB)", min_value=250, max_value=4000, value=2000, step=50)
sc_h = st.sidebar.number_input("Screen Height (cm)", min_value=5, max_value=20, value=12, step=1)
sc_w = st.sidebar.number_input("Screen Width (cm)", min_value=0, max_value=20, value=6, step=1)
talk_time = st.sidebar.number_input("Talk Time (hours)", min_value=2, max_value=20, value=10, step=1)
three_g = st.sidebar.selectbox("3G", ["Yes", "No"])
touch_screen = st.sidebar.selectbox("Touch Screen", ["Yes", "No"])
wifi = st.sidebar.selectbox("WiFi", ["Yes", "No"])

# -----------------------------
# Feature engineering helpers
# -----------------------------
def yes_no(v):
    return 1 if v == "Yes" else 0

def build_feature_vector():
    data = {
        'battery_power': battery_power,
        'blue': yes_no(blue),
        'clock_speed': clock_speed,
        'dual_sim': yes_no(dual_sim),
        'four_g': yes_no(four_g),
        'int_memory': int_memory,
        'm_dep': m_dep,
        'n_cores': n_cores,
        'pc': pc,
        'px_height': px_height,
        'px_width': px_width,
        'ram': ram,
        'sc_h': sc_h,
        'sc_w': sc_w,
        'talk_time': talk_time,
        'three_g': yes_no(three_g),
        'touch_screen': yes_no(touch_screen),
        'wifi': yes_no(wifi),
        'mobile_wt_High': 1 if mobile_wt == "High" else 0,
        'mobile_wt_Low': 1 if mobile_wt == "Low" else 0,
        'mobile_wt_Med': 1 if mobile_wt == "Med" else 0,
        'fc_log': np.log1p(fc),
    }

    X_input = pd.DataFrame([data])
    X_input = X_input.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    return X_input

# -----------------------------
# Prediction Section
# -----------------------------
st.subheader("📊 Prediction Summary")

if st.button("Predict Price Range"):

    X_input = build_feature_vector()
    X_model_input = scaler.transform(X_input) if USES_SCALING else X_input

    pred = model.predict(X_model_input)[0]

    st.success(
        f"📱 Estimated Price Category: **{PRICE_LABELS.get(pred, 'Unknown')}**"
    )

    st.divider()

    # -----------------------------
    # Confidence Scores
    # -----------------------------
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_model_input)[0]

        proba_df = pd.DataFrame({
            "Price Range": [PRICE_LABELS[i] for i in range(len(proba))],
            "Probability": proba
        })

        st.subheader("📈 Confidence Scores")
        st.bar_chart(proba_df.set_index("Price Range"))

        st.divider()

    # -----------------------------
    # Input Summary (TABLE AT BOTTOM)
    # -----------------------------
    st.subheader("📋 Feature Summary")
    st.caption("These are the processed features that were sent to the trained model for prediction.")
    st.dataframe(X_input, use_container_width=True)

else:
    st.info("Fill the details on the left and click **Predict Price Range** to generate a prediction.")

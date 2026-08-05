import streamlit as st
import pandas as pd
import joblib

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------
st.set_page_config(
    page_title="Mobile Price Range Prediction Demo",
    page_icon="📱",
    layout="wide"
)

# -------------------------------------------------------
# Load Model Artifacts
# -------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    needs_scaling = joblib.load("needs_scaling.pkl")
    return model, scaler, feature_cols, needs_scaling

model, scaler, feature_cols, needs_scaling = load_artifacts()

# -------------------------------------------------------
# Labels
# -------------------------------------------------------
PRICE_LABELS = {
    0: "Low Cost",
    1: "Medium Cost",
    2: "High Cost",
    3: "Very High Cost"
}

# -------------------------------------------------------
# Title
# -------------------------------------------------------
st.title("📱 Mobile Price Range Prediction Demo")

st.write(
    f"**Model in use:** {type(model).__name__}. "
    "Enter the mobile specifications on the left and click "
    "**Predict** to see the estimated price range."
)

# -------------------------------------------------------
# Sidebar Inputs
# -------------------------------------------------------
st.sidebar.header("📋 Mobile Specifications")

battery_power = st.sidebar.number_input(
    "Battery Power (mAh)", 500, 2000, 1200
)

clock_speed = st.sidebar.number_input(
    "Clock Speed (GHz)", 0.5, 3.0, 1.5, step=0.1
)

fc = st.sidebar.number_input(
    "Front Camera (MP)", 0, 20, 5
)

int_memory = st.sidebar.number_input(
    "Internal Memory (GB)", 2, 64, 32
)

m_dep = st.sidebar.number_input(
    "Mobile Depth (cm)", 0.1, 1.0, 0.5, step=0.1
)

mobile_wt = st.sidebar.selectbox(
    "Mobile Weight Category",
    ["Low", "Med", "High"]
)

n_cores = st.sidebar.number_input(
    "Number of Cores", 1, 8, 4
)

pc = st.sidebar.number_input(
    "Primary Camera (MP)", 0, 20, 10
)

px_height = st.sidebar.number_input(
    "Pixel Height", 0, 2000, 600
)

px_width = st.sidebar.number_input(
    "Pixel Width", 500, 2000, 1200
)

ram = st.sidebar.number_input(
    "RAM (MB)", 256, 4000, 2000
)

sc_h = st.sidebar.number_input(
    "Screen Height (cm)", 5, 20, 12
)

sc_w = st.sidebar.number_input(
    "Screen Width (cm)", 0, 18, 6
)

talk_time = st.sidebar.number_input(
    "Talk Time (hours)", 2, 20, 10
)

blue = st.sidebar.selectbox(
    "Bluetooth",
    ["Yes", "No"]
)

dual_sim = st.sidebar.selectbox(
    "Dual SIM",
    ["Yes", "No"]
)

four_g = st.sidebar.selectbox(
    "4G",
    ["Yes", "No"]
)

three_g = st.sidebar.selectbox(
    "3G",
    ["Yes", "No"]
)

touch_screen = st.sidebar.selectbox(
    "Touch Screen",
    ["Yes", "No"]
)

wifi = st.sidebar.selectbox(
    "WiFi",
    ["Yes", "No"]
)

# -------------------------------------------------------
# Prediction Section
# -------------------------------------------------------
st.subheader("Prediction")

predict = st.button("Predict Price Range")

if predict:

    yn = {"Yes": 1, "No": 0}
    wt_map = {"Low": 0, "Med": 1, "High": 2}

    row = {
        "battery_power": battery_power,
        "blue": yn[blue],
        "clock_speed": clock_speed,
        "dual_sim": yn[dual_sim],
        "fc": fc,
        "four_g": yn[four_g],
        "int_memory": int_memory,
        "m_dep": m_dep,
        "mobile_wt": wt_map[mobile_wt],
        "n_cores": n_cores,
        "pc": pc,
        "px_height": px_height,
        "px_width": px_width,
        "ram": ram,
        "sc_h": sc_h,
        "sc_w": sc_w,
        "talk_time": talk_time,
        "three_g": yn[three_g],
        "touch_screen": yn[touch_screen],
        "wifi": yn[wifi],
        "px_area": px_height * px_width,
        "screen_area": sc_h * sc_w,
    }

    X_new = pd.DataFrame([row])[feature_cols]

    if needs_scaling:
        X_model = pd.DataFrame(
            scaler.transform(X_new),
            columns=feature_cols
        )
    else:
        X_model = X_new

    pred = model.predict(X_model)[0]

    st.success(
        f"📱 Predicted Price Range: **{pred} - {PRICE_LABELS[pred]}**"
    )

    # ---------------------------------------------------
    # Probability Chart
    # ---------------------------------------------------
    if hasattr(model, "predict_proba"):

        proba = model.predict_proba(X_model)[0]

        proba_df = pd.DataFrame({
            "Price Range": [
                "0 - Low Cost",
                "1 - Medium Cost",
                "2 - High Cost",
                "3 - Very High Cost"
            ],
            "Probability": proba
        })

        st.subheader("Class Probabilities")

        st.bar_chart(
            proba_df.set_index("Price Range")
        )

    # ---------------------------------------------------
    # Display Input Features
    # ---------------------------------------------------
    st.subheader("Input Features Sent to Model")
    st.dataframe(X_new)

else:

    st.info(
        "Fill the details on the left and click **Predict Price Range**."
    )

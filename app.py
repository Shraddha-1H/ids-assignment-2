import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Mobile Price Range Predictor",
    page_icon="📱",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    needs_scaling = joblib.load("needs_scaling.pkl")
    return model, scaler, feature_cols, needs_scaling

model, scaler, feature_cols, needs_scaling = load_artifacts()

st.title("📱 Mobile Price Range Predictor")

st.write(
    "Enter the phone specifications below to predict its price range "
    "(0 = Low, 1 = Medium, 2 = High, 3 = Very High)."
)

with st.form("spec_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        battery_power = st.number_input("Battery Power (mAh)", 500, 2000, 1200)
        clock_speed = st.number_input("Clock Speed (GHz)", 0.5, 3.0, 1.5, step=0.1)
        fc = st.number_input("Front Camera (MP)", 0, 20, 5)
        int_memory = st.number_input("Internal Memory (GB)", 2, 64, 32)
        m_dep = st.number_input("Mobile Depth (cm)", 0.1, 1.0, 0.5, step=0.1)
        mobile_wt = st.selectbox("Mobile Weight Category", ["Low", "Med", "High"])

    with col2:
        n_cores = st.number_input("Number of Cores", 1, 8, 4)
        pc = st.number_input("Primary Camera (MP)", 0, 20, 10)
        px_height = st.number_input("Pixel Height", 0, 2000, 600)
        px_width = st.number_input("Pixel Width", 500, 2000, 1200)
        ram = st.number_input("RAM (MB)", 256, 4000, 2000)
        sc_h = st.number_input("Screen Height (cm)", 5, 20, 12)

    with col3:
        sc_w = st.number_input("Screen Width (cm)", 0, 18, 6)
        talk_time = st.number_input("Talk Time (hours)", 2, 20, 10)
        blue = st.selectbox("Bluetooth", ["Yes", "No"])
        dual_sim = st.selectbox("Dual SIM", ["Yes", "No"])
        four_g = st.selectbox("4G", ["Yes", "No"])
        three_g = st.selectbox("3G", ["Yes", "No"])
        touch_screen = st.selectbox("Touch Screen", ["Yes", "No"])
        wifi = st.selectbox("WiFi", ["Yes", "No"])

    submitted = st.form_submit_button("Predict Price Range")

if submitted:

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

    # Save original values for display
    input_df = pd.DataFrame([row])

    # Reorder columns for prediction
    X_new = input_df[feature_cols]

    if needs_scaling:
        X_model = pd.DataFrame(
            scaler.transform(X_new),
            columns=feature_cols
        )
    else:
        X_model = X_new

    pred = model.predict(X_model)[0]

    label_map = {
        0: "Low (0)",
        1: "Medium (1)",
        2: "High (2)",
        3: "Very High (3)"
    }

    st.success(f"📱 Predicted Price Range: **{label_map[pred]}**")

    if hasattr(model, "predict_proba"):

        st.subheader("Class Probabilities")

        proba = model.predict_proba(X_model)[0]

        st.bar_chart(
            pd.Series(
                proba,
                index=[label_map[i] for i in model.classes_]
            )
        )

    st.divider()

    st.subheader("Input Features Sent to Model")

    st.dataframe(
        input_df,
        use_container_width=True,
        hide_index=True
    )

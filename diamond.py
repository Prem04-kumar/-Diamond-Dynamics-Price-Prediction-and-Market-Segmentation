import streamlit as st
import pandas as pd
import joblib
import json

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Diamond Dynamics", layout="wide")

# ---------------- LOAD MODELS ----------------
price_model = joblib.load(r"C:\VSCODE\model.pkl")
cluster_model = joblib.load(r"C:\VSCODE\cluster_model.pkl")
scaler_cluster = joblib.load(r"C:\VSCODE\scaler.pkl")
encoder = joblib.load(r"C:\VSCODE\encoder.pkl")

with open(r"C:\VSCODE\cluster_names.json", "r") as f:
    cluster_names = json.load(f)

# ---------------- HEADER ----------------
st.markdown(
    """
    <h1 style='text-align: center; color: #4B8BBE;'>💎 Diamond Dynamics</h1>
    <h4 style='text-align: center;'>Price Prediction & Market Segmentation</h4>
    """,
    unsafe_allow_html=True
)

st.write("Enter diamond details in the sidebar to get predictions.")

st.sidebar.subheader("💡 Try Sample Diamonds")

preset_options = {
    "💍 Budget Small Diamond": {
        "carat": 0.3, "depth": 62.0, "table": 55.0,
        "x": 4.2, "y": 4.1, "z": 2.5,
        "cut": "Good", "color": "H", "clarity": "SI2"
    },
    "✨ Everyday Wear Diamond": {
        "carat": 0.6, "depth": 61.5, "table": 56.0,
        "x": 5.5, "y": 5.4, "z": 3.2,
        "cut": "Very Good", "color": "G", "clarity": "SI1"
    },
    "🎁 Mid-Range Gift Diamond": {
        "carat": 1.0, "depth": 62.3, "table": 57.0,
        "x": 6.5, "y": 6.4, "z": 4.0,
        "cut": "Premium", "color": "F", "clarity": "VS2"
    },
    "💎 Premium Engagement Diamond": {
        "carat": 1.8, "depth": 61.8, "table": 58.0,
        "x": 7.8, "y": 7.7, "z": 4.8,
        "cut": "Ideal", "color": "E", "clarity": "VVS2"
    },
    "👑 Luxury Showcase Diamond": {
        "carat": 2.5, "depth": 62.1, "table": 59.0,
        "x": 8.8, "y": 8.7, "z": 5.5,
        "cut": "Ideal", "color": "D", "clarity": "IF"
    }
}

selected_preset = st.sidebar.selectbox(
    "Choose a preset",
    ["Custom Input"] + list(preset_options.keys())
)

if selected_preset != "Custom Input":
    preset = preset_options[selected_preset]

    carat = st.sidebar.number_input("Carat", value=preset["carat"])
    depth = st.sidebar.number_input("Depth (%)", value=preset["depth"])
    table = st.sidebar.number_input("Table (%)", value=preset["table"])

    x = st.sidebar.number_input("Length (mm)", value=preset["x"])
    y = st.sidebar.number_input("Width (mm)", value=preset["y"])
    z = st.sidebar.number_input("Height (mm)", value=preset["z"])

    cut = st.sidebar.selectbox(
        "Cut",
        ['Fair','Good','Very Good','Premium','Ideal'],
        index=['Fair','Good','Very Good','Premium','Ideal'].index(preset["cut"])
    )

    color = st.sidebar.selectbox(
        "Color",
        ['J','I','H','G','F','E','D'],
        index=['J','I','H','G','F','E','D'].index(preset["color"])
    )

    clarity = st.sidebar.selectbox(
        "Clarity",
        ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'],
        index=['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'].index(preset["clarity"])
    )

else:
    carat = st.sidebar.number_input("Carat", 0.1, 5.0, 1.0)
    depth = st.sidebar.number_input("Depth (%)", 40.0, 80.0, 61.5)
    table = st.sidebar.number_input("Table (%)", 40.0, 80.0, 55.0)

    x = st.sidebar.number_input("Length (mm)", 0.1, 10.0, 5.0)
    y = st.sidebar.number_input("Width (mm)", 0.1, 10.0, 5.0)
    z = st.sidebar.number_input("Height (mm)", 0.1, 10.0, 3.0)

    cut = st.sidebar.selectbox("Cut", ['Fair','Good','Very Good','Premium','Ideal'])
    color = st.sidebar.selectbox("Color", ['J','I','H','G','F','E','D'])
    clarity = st.sidebar.selectbox("Clarity", ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'])

predict_btn = st.sidebar.button("Predict")

# ---------------- PREDICTION ----------------
if predict_btn:

    input_df = pd.DataFrame([{
        "carat": carat,
        "cut": cut,
        "color": color,
        "clarity": clarity,
        "depth": depth,
        "table": table,
        "x": x,
        "y": y,
        "z": z
    }])

    cat_cols = ["cut", "color", "clarity"]
    input_df[cat_cols] = encoder.transform(input_df[cat_cols])

    price = price_model.predict(input_df)[0]

    cluster = cluster_model.predict(scaler_cluster.transform(input_df))[0]
    cluster_label = cluster_names[str(cluster)]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div style="padding:20px; border-radius:10px; background-color:#E8F6F3;">
                <h3 style="color:#117864;">💰 Estimated Price</h3>
                <h2 style="color:#0B5345;">₹ {int(price):,}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div style="padding:20px; border-radius:10px; background-color:#FEF5E7;">
                <h3 style="color:#B9770E;">📦 Market Segment</h3>
                <h2 style="color:#7D6608;">{cluster_label}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

else:
    st.info("👈 Enter values in the sidebar and click Predict")
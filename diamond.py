"""
Diamond Dynamics - Streamlit Application
Modules: 1) Price Prediction  2) Market Segment Prediction
"""
 
import streamlit as st
import pandas as pd
import numpy as np
import joblib
 
# ========================================
# PAGE CONFIGURATION
# ========================================
st.set_page_config(
    page_title="💎 Diamond Dynamics",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# ========================================
# LOAD MODELS
# ========================================
@st.cache_resource
def load_models():
    """Load all saved models and scalers"""
    try:
        # Load regression model (NO SCALER for regression)
        regression_model = joblib.load('best_model_diamond.pkl')
        
        # Load clustering model, scaler, and cluster names
        clustering_model = joblib.load('kmeans_model.pkl')
        scaler_cluster = joblib.load('scaler_cluster.pkl')
        cluster_names = joblib.load('cluster_names.pkl')
        
        return regression_model, clustering_model, scaler_cluster, cluster_names
    
    except FileNotFoundError as e:
        st.error(f"❌ Error loading models: {e}")
        st.info("Please ensure all model files are in the same directory as this script.")
        st.stop()
 
# Load all models
regression_model, clustering_model, scaler_cluster, cluster_names = load_models()
 
# ========================================
# HELPER FUNCTIONS
# ========================================
def calculate_volume(x, y, z):
    """Calculate diamond volume"""
    return x * y * z
 
def calculate_dimension_ratio(x, y, z):
    """Calculate dimension ratio"""
    if z == 0:
        return 0
    return (x + y) / (2 * z)
 
# ========================================
# HEADER
# ========================================
st.title("💎 Diamond Dynamics: Price Prediction & Market Segmentation")
st.markdown("---")
st.markdown("""
Welcome to the **Diamond Dynamics** application! This tool helps you:
- 💰 **Predict diamond prices** based on characteristics
- 🎯 **Identify market segments** for diamonds
""")
st.markdown("---")
 
# ========================================
# SIDEBAR - INPUT FORM
# ========================================
st.sidebar.header("📝 Diamond Specifications")
st.sidebar.markdown("Enter the diamond characteristics below:")
 
# Numerical Inputs
carat = st.sidebar.number_input(
    "Carat Weight",
    min_value=0.1,
    max_value=10.0,
    value=1.0,
    step=0.01,
    help="Weight of the diamond in carats"
)
 
depth = st.sidebar.number_input(
    "Depth %",
    min_value=40.0,
    max_value=80.0,
    value=60.0,
    step=0.1,
    help="Total depth percentage"
)
 
table = st.sidebar.number_input(
    "Table %",
    min_value=40.0,
    max_value=80.0,
    value=57.0,
    step=0.1,
    help="Width of top facet"
)
 
x = st.sidebar.number_input(
    "Length (x) in mm",
    min_value=0.0,
    max_value=20.0,
    value=5.0,
    step=0.01,
    help="Length of diamond in mm"
)
 
y = st.sidebar.number_input(
    "Width (y) in mm",
    min_value=0.0,
    max_value=20.0,
    value=5.0,
    step=0.01,
    help="Width of diamond in mm"
)
 
z = st.sidebar.number_input(
    "Depth (z) in mm",
    min_value=0.0,
    max_value=20.0,
    value=3.0,
    step=0.01,
    help="Depth of diamond in mm"
)
 
# Categorical Inputs (with encoded values matching OrdinalEncoder)
st.sidebar.markdown("---")
st.sidebar.subheader("Quality Attributes")
 
# NOTE: These mappings match OrdinalEncoder with 0-indexing
cut_mapping = {
    'Fair': 0,
    'Good': 1,
    'Very Good': 2,
    'Premium': 3,
    'Ideal': 4
}
cut_display = st.sidebar.selectbox(
    "Cut Quality",
    options=list(cut_mapping.keys()),
    index=4,  # Default to Ideal
    help="Quality of the cut"
)
cut = cut_mapping[cut_display]
 
color_mapping = {
    'J': 0,
    'I': 1,
    'H': 2,
    'G': 3,
    'F': 4,
    'E': 5,
    'D': 6
}
color_display = st.sidebar.selectbox(
    "Color Grade",
    options=list(color_mapping.keys()),
    index=6,  # Default to D
    help="Color grade from J (worst) to D (best)"
)
color = color_mapping[color_display]
 
clarity_mapping = {
    'I1': 0,
    'SI2': 1,
    'SI1': 2,
    'VS2': 3,
    'VS1': 4,
    'VVS2': 5,
    'VVS1': 6,
    'IF': 7
}
clarity_display = st.sidebar.selectbox(
    "Clarity",
    options=list(clarity_mapping.keys()),
    index=7,  # Default to IF
    help="Clarity grade from I1 (worst) to IF (best)"
)
clarity = clarity_mapping[clarity_display]
 
# Calculate derived features
volume = calculate_volume(x, y, z)
dimension_ratio = calculate_dimension_ratio(x, y, z)
 
st.sidebar.markdown("---")
st.sidebar.info(f"""
**Calculated Features:**
- Volume: {volume:.2f} mm³
- Dimension Ratio: {dimension_ratio:.4f}
""")
 
# ========================================
# MAIN CONTENT - TWO COLUMNS
# ========================================
col1, col2 = st.columns(2)
 
# ========================================
# MODULE 1: PRICE PREDICTION
# ========================================
with col1:
    st.header("💰 Price Prediction")
    st.markdown("Predict the price of the diamond based on its characteristics.")
    
    if st.button("🔮 Predict Price", use_container_width=True):
        try:
            # Prepare input data with correct feature order
            # Features: volume, y, clarity, color, carat
            input_data_regression = pd.DataFrame({
                'volume': [volume],
                'y': [y],
                'clarity': [clarity],
                'color': [color],
                'carat': [carat]
            })
 
            # Ensure correct column order
            input_data_regression = input_data_regression[
                ['volume', 'y', 'clarity', 'color', 'carat']
            ]
 
            # Make prediction (no scaling needed)
            predicted_price = regression_model.predict(input_data_regression)[0]
 
            # Display results
            st.success("✅ Prediction Complete!")
            st.metric(
                label="Predicted Price",
                value=f"₹{predicted_price:,.2f}",
            )
 
            # Additional insights
            st.markdown("---")
            st.markdown("**Diamond Summary:**")
            st.write(f"- **Carat:** {carat}")
            st.write(f"- **Color:** {color_display}")
            st.write(f"- **Clarity:** {clarity_display}")
            st.write(f"- **Volume:** {volume:.2f} mm³")
            st.write(f"- **Width (y):** {y} mm")
 
        except Exception as e:
            st.error(f"❌ Error making prediction: {e}")
            st.error(f"Details: {str(e)}")
 
# ========================================
# MODULE 2: MARKET SEGMENT PREDICTION
# ========================================
with col2:
    st.header("🎯 Market Segment Prediction")
    st.markdown("Identify which market category this diamond belongs to.")
    
    if st.button("🔍 Predict Cluster", use_container_width=True):
        try:
            # Prepare input data for clustering
            # Features: carat, cut, color, clarity, depth, table, volume, dimension_ratio
            input_data_cluster = pd.DataFrame({
                'carat': [carat],
                'cut': [cut],
                'color': [color],
                'clarity': [clarity],
                'depth': [depth],
                'table': [table],
                'volume': [volume],
                'dimension_ratio': [dimension_ratio]
            })
            
            # Scale the input for clustering
            input_scaled = scaler_cluster.transform(input_data_cluster)
            
            # Predict cluster
            cluster_id = clustering_model.predict(input_scaled)[0]
            cluster_name = cluster_names[cluster_id]
            
            # Display results
            st.success("✅ Segment Identified!")
            st.metric(
                label="Market Segment",
                value=cluster_name,
                delta=None
            )
            
            # Additional cluster info
            st.markdown("---")
            st.markdown(f"**Cluster ID:** {cluster_id}")
            
            # Display cluster characteristics
            st.info(f"""
            This diamond belongs to the **{cluster_name}** category.
            
            These diamonds typically have:
            - Similar carat weights
            - Comparable quality characteristics
            - Similar price ranges
            """)
            
        except Exception as e:
            st.error(f"❌ Error making prediction: {e}")
            st.error(f"Details: {str(e)}")
 
# ========================================
# FOOTER
# ========================================
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>💎 <b>Diamond Dynamics</b> | Price Prediction & Market Segmentation</p>
    <p style='font-size: 12px; color: gray;'>Built with Streamlit | Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)
 
# ========================================
# SIDEBAR FOOTER
# ========================================
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📊 About
This application uses:
- **Random Forest Model** for price prediction
- **K-Means Clustering** for market segmentation
 
### 📝 Instructions
1. Enter diamond specifications
2. Click prediction buttons
3. View results in real-time
 
### ⚙️ Technical Details
**Price Prediction Features:**
- Volume, Width (y), Clarity, Color, Carat
 
**Clustering Features:**
- Carat, Cut, Color, Clarity, Depth, Table, Volume, Dimension Ratio
""")
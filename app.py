import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os
import base64

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Telewire Analytics", layout="wide", page_icon="📡")

# --- 2. GLOBAL STYLING (The "Glassmorphism" UI) ---
def apply_ui_design():
    st.markdown("""
    <style>
    /* Main Background: Deep Gradient */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #ffffff;
    }
    
    /* Sidebar: Frosted Glass Effect */
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(15px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Sidebar Text & Titles */
    [data-testid="stSidebar"] .stMarkdown h1, [data-testid="stSidebar"] .stMarkdown h2 {
        color: #00D4FF !important;
        text-shadow: 0px 0px 10px rgba(0, 212, 255, 0.5);
    }

    /* Metric Cards: Glowing Glass */
    div.stMetric {
        background: rgba(255, 255, 255, 0.07);
        border-radius: 15px;
        padding: 15px !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    /* Interactive Hover for Cards */
    div.stMetric:hover {
        border-color: #00D4FF;
        transform: translateY(-3px);
        transition: 0.3s ease;
    }

    /* Chart Containers */
    .stPlotlyChart {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 20px;
        padding: 10px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Form & Button Styling */
    .stButton>button {
        background: linear-gradient(45deg, #00D4FF, #0056b3);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: bold;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

apply_ui_design()

# --- 3. ASSET LOADING ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(BASE_DIR, 'models')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Telco-Customer-Churn.csv')

@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(os.path.join(MODELS_PATH, 'churn_model.pkl'))
        encoders = joblib.load(os.path.join(MODELS_PATH, 'encoders.pkl'))
        features = joblib.load(os.path.join(MODELS_PATH, 'feature_columns.pkl'))
        return model, encoders, features
    except: return None, None, None

@st.cache_data
def get_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
        return df
    return pd.DataFrame()

model, encoders, feature_names = load_artifacts()
df = get_data()

# --- 4. SIDEBAR NAVIGATION & FILTERS ---
with st.sidebar:
    st.markdown("# 🚀 Telewire AI")
    st.markdown("### Command Center")
    page = st.radio("SELECT VIEW", ["Market Dashboard", "Individual Predictor", "Batch Processing"])
    
    st.divider()

    # --- GLOBAL FILTERS SECTION ---
    st.markdown("### ⚙️ Global Filters")
    if not df.empty:
        contract_opt = st.multiselect(
            "Contract Type", 
            options=df["Contract"].unique(), 
            default=df["Contract"].unique()
        )
        internet_opt = st.multiselect(
            "Internet Technology", 
            options=df["InternetService"].unique(), 
            default=df["InternetService"].unique()
        )
        
        # Apply filter logic
        filtered_df = df[
            (df["Contract"].isin(contract_opt)) & 
            (df["InternetService"].isin(internet_opt))
        ]
    else:
        filtered_df = df

    st.divider()
    
    st.markdown("### 🖥️ System Status")
    status = "🟢 Online" if model else "🔴 Offline"
    st.info(f"Model Status: {status}")

# --- 5. PAGE: MARKET DASHBOARD ---
if page == "Market Dashboard":
    st.title("📊 Strategic Insights Dashboard")
    
    if filtered_df.empty:
        st.warning("No data available based on current filters. Please adjust selections in the sidebar.")
    else:
        # Vibrant Color Map
        color_map = {"No": "#00D4FF", "Yes": "#FF4B4B"}
        
        # Row 1: Metrics (Using filtered_df)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Base", len(filtered_df))
        m2.metric("Churn Rate", f"{(filtered_df['Churn']=='Yes').mean():.1%}")
        m3.metric("Avg Tenure", f"{filtered_df['tenure'].mean():.1f} mo")
        m4.metric("Avg Bill", f"${filtered_df['MonthlyCharges'].mean():.2f}")

        st.divider()

        # Row 2: Contract & Tenure (Using filtered_df)
        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.histogram(filtered_df, x="Contract", color="Churn", barmode="group",
                               template="plotly_dark", title="<b>Churn by Contract</b>",
                               color_discrete_map=color_map)
            fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig1, use_container_width=True)
            
        with c2:
            fig2 = px.scatter(filtered_df, x="tenure", y="MonthlyCharges", color="Churn",
                             opacity=0.6, template="plotly_dark", title="<b>Charges vs Tenure</b>",
                             color_discrete_map=color_map)
            fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig2, use_container_width=True)

        # Row 3: Payment & Distribution (Using filtered_df)
        c3, c4 = st.columns(2)
        with c3:
            fig3 = px.pie(filtered_df, names="PaymentMethod", hole=0.5, template="plotly_dark",
                         title="<b>Payment Distribution</b>", color_discrete_sequence=px.colors.qualitative.Prism)
            fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig3, use_container_width=True)
            
        with c4:
            fig4 = px.box(filtered_df, x="Churn", y="MonthlyCharges", color="Churn",
                         template="plotly_dark", title="<b>Charge Dispersion</b>",
                         color_discrete_map=color_map, notched=True)
            fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig4, use_container_width=True)

        # Row 4: Feature Importance (Calculated from model, displayed here)
        st.divider()
        if model:
            st.subheader("🚀 Machine Learning Drivers")
            fi = pd.DataFrame({'Feature': feature_names, 'Impact': model.feature_importances_}).sort_values('Impact', ascending=False).head(10)
            fig_fi = px.bar(fi, x='Impact', y='Feature', orientation='h', template="plotly_dark",
                           color='Impact', color_continuous_scale='Plasma', title="<b>Top Drivers of Churn</b>")
            fig_fi.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_fi, use_container_width=True)

# --- 6. PAGE: INDIVIDUAL PREDICTOR ---
elif page == "Individual Predictor":
    st.title("🔮 Predictive AI Terminal")
    with st.form("input_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            tenure = st.slider("Tenure (Months)", 0, 72, 24)
        with c2:
            st.markdown("#### Service Plan")
            internet = st.selectbox("Internet", ["DSL", "Fiber optic", "No"])
            security = st.selectbox("Security", ["Yes", "No", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        with c3:
            st.markdown("#### Financials")
            monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
            total = st.number_input("Total Charges", 0.0, 10000.0, 500.0)
            payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
            billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            
        if st.form_submit_button("🔥 ANALYZE RISK"):
            input_dict = {
                'gender': gender, 'SeniorCitizen': senior, 'Partner': partner, 'Dependents': 'No',
                'tenure': tenure, 'PhoneService': 'Yes', 'MultipleLines': 'No', 
                'InternetService': internet, 'OnlineSecurity': security, 'OnlineBackup': 'No',
                'DeviceProtection': 'No', 'TechSupport': support, 'StreamingTV': 'No',
                'StreamingMovies': 'No', 'Contract': contract, 'PaperlessBilling': billing,
                'PaymentMethod': payment, 'MonthlyCharges': monthly, 'TotalCharges': total
            }
            input_df = pd.DataFrame([input_dict])
            for col, enc in encoders.items():
                if col in input_df.columns: input_df[col] = enc.transform(input_df[col].astype(str))
            
            prob = model.predict_proba(input_df[feature_names])[0][1]
            st.divider()
            if prob > 0.5:
                st.error(f"### ALERT: High Churn Probability: {prob:.1%}")
            else:
                st.success(f"### Customer Stable. Risk Score: {prob:.1%}")

# --- 7. PAGE: BATCH PROCESSING ---
elif page == "Batch Processing":
    st.title("📁 Batch Analysis Terminal")
    up = st.file_uploader("Upload CSV Data", type="csv")
    if up and model:
        raw = pd.read_csv(up)
        # Preprocessing
        batch = raw.copy()
        batch['TotalCharges'] = pd.to_numeric(batch['TotalCharges'], errors='coerce').fillna(0)
        for col, enc in encoders.items():
            if col in batch.columns: batch[col] = enc.transform(batch[col].astype(str))
        
        raw['Churn_Risk'] = model.predict_proba(batch[feature_names])[:, 1]
        st.dataframe(raw.style.background_gradient(subset=['Churn_Risk'], cmap='RdYlGn_r'))
        st.download_button("📥 Export Analysis", raw.to_csv(index=False), "telewire_analysis.csv")
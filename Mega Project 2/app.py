# app.py
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

st.set_page_config(page_title="Placement Predictor", layout="wide")
st.title("🎓 Campus Placement Prediction Using Machine Learning ")

# -----------------------------
# Helpers
# -----------------------------
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

@st.cache_data
def load_data(uploaded):
    return pd.read_csv(uploaded)

@st.cache_data
def load_models():
    models = {}
    for name in ["placement_pipe", "salary_reg", "role_clf", "le_jobrole"]:
        path = os.path.join(ARTIFACT_DIR, f"{name}.pkl")
        models[name] = joblib.load(path) if os.path.exists(path) else None
    return models

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("📂 Upload CSV (must include a 'Placed' or 'Placement' column)", type=["csv"])
if uploaded_file is None:
    st.info("Upload your dataset CSV to continue. Or use generate_dataset.py to create one.")
    st.stop()

raw = load_data(uploaded_file)

# Normalize column names
raw.columns = [c.strip() for c in raw.columns]
if "Placed" not in raw.columns and "Placement" in raw.columns:
    raw = raw.rename(columns={"Placement": "Placed"})

models = load_models()

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dataset & EDA", 
    "🤖 Model Training & Comparison", 
    "🔮 Prediction", 
    "📘 Docs & Artifacts"
])

# ==========================================================
# TAB 1: DATASET & EDA
# ==========================================================
with tab1:
    st.header("📊 Dataset Preview & Exploratory Data Analysis")

    # Dataset preview
    st.subheader("🔍 Preview Dataset")
    st.dataframe(raw.head(20), use_container_width=True)

    # KPIs
    st.subheader("📌 Key Metrics")
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Total Students", len(raw))
    with col2: 
        if "Placed" in raw.columns:
            st.metric("Placement Rate", f"{raw['Placed'].mean()*100:.1f}%")
    with col3:
        if "Salary" in raw.columns and raw["Salary"].notna().sum() > 0:
            st.metric("Avg Salary (₹)", f"{raw['Salary'].mean():,.0f}")

    st.markdown("---")

    # Placement distribution
    if "Placed" in raw.columns:
        st.subheader("🎓 Placement Distribution")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(raw, names="Placed", title="Placement Split", 
                         color="Placed", color_discrete_map={0:"red",1:"green"})
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.histogram(raw, x="Placed", color="Placed", 
                               text_auto=True, title="Placement Count")
            st.plotly_chart(fig, use_container_width=True)

    # Salary analysis
    if "Salary" in raw.columns and raw["Salary"].notna().sum() > 0:
        st.subheader("💰 Salary Analysis")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.box(raw, y="Salary", color="Placed", 
                         title="Salary Distribution", 
                         color_discrete_map={0:"red",1:"green"})
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            if "Branch" in raw.columns:
                fig = px.violin(raw, x="Branch", y="Salary", box=True, points="all",
                                title="Branch vs Salary", color="Branch")
                st.plotly_chart(fig, use_container_width=True)

    # CGPA vs Placement
    if "CGPA" in raw.columns and "Placed" in raw.columns:
        st.subheader("📈 CGPA vs Placement & Salary")
        fig = px.scatter(raw, x="CGPA", y="Salary" if "Salary" in raw.columns else "CGPA",
                         color="Placed", size="CGPA", hover_data=["Branch","Gender"],
                         title="Impact of CGPA on Placement & Salary")
        st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    st.subheader("🔗 Correlation Heatmap")
    num = raw.select_dtypes(include=[np.number])
    if num.shape[1] > 1:
        fig = px.imshow(num.corr(), text_auto=True, title="Feature Correlation Heatmap",
                        color_continuous_scale="Tealgrn")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough numeric columns for correlation heatmap.")

# ==========================================================
# TAB 2: MODEL TRAINING & COMPARISON
# ==========================================================
with tab2:
    st.header("🤖 Model Training & Comparison")
    st.write("Models are trained via `train_models.py`. If artifacts are available, they are loaded below:")

    st.subheader("📂 Loaded Artifacts")
    for k, v in models.items():
        st.write(f"{k}: {'✅ Loaded' if v is not None else '❌ Not found'}")

    if models.get("placement_pipe"):
        st.success("Placement model is available.")
        st.text(str(models["placement_pipe"]))
    else:
        st.warning("Run `python train_models.py` to generate models.")

# ==========================================================
# TAB 3: PREDICTION
# ==========================================================
with tab3:
    st.header("🔮 Predict for a Student")

    # Option 1: Predict by Roll Number
    st.subheader("📌 Search by Roll Number")
    if "StudentID" in raw.columns:
        roll_input = st.text_input("Enter Student Roll Number (e.g., S0001)")
        if st.button("🔍 Predict by Roll Number"):
            student_row = raw[raw["StudentID"] == roll_input]
            if not student_row.empty:
                st.write("✅ Student Found:", student_row)

                input_df = student_row.drop(columns=["Placed","Salary","JobRole","Placement"], errors="ignore")

                # Placement prediction
                if models.get("placement_pipe"):
                    proba = models["placement_pipe"].predict_proba(input_df)[0][1]
                    st.metric("Placement Probability", f"{proba*100:.2f}%")
                    if proba >= 0.5:
                        st.success("🎉 Likely to be PLACED")
                    else:
                        st.error("❌ Likely NOT placed")

                # Salary prediction
                if models.get("salary_reg"):
                    try:
                        sal = models["salary_reg"].predict(input_df)[0]
                        st.info(f"💰 Estimated Salary: ₹{int(sal):,} (≈ {sal/100000:.2f} LPA)")
                    except Exception as e:
                        st.warning(f"Salary prediction skipped: {e}")

                # Job Role prediction
                if models.get("role_clf") and models.get("le_jobrole"):
                    try:
                        role_enc = models["role_clf"].predict(input_df)[0]
                        role = models["le_jobrole"].inverse_transform([int(role_enc)])[0]
                        st.info(f"👔 Predicted Job Role: {role}")
                    except Exception as e:
                        st.warning(f"Job role prediction skipped: {e}")
            else:
                st.error("❌ No student found with that Roll Number.")

    st.markdown("---")

    # Option 2: Predict Manually (existing form)
    st.subheader("✍️ Manual Entry")
    feature_cols = [c for c in raw.columns if c not in ["Placed","Salary","JobRole","Placement","StudentID"]]
    form = st.form("predict_form")
    inputs = {}
    for c in feature_cols:
        if raw[c].dtype == object or raw[c].nunique() <= 20:
            inputs[c] = form.selectbox(c, raw[c].astype(str).unique().tolist())
        else:
            mn, mx, mean = raw[c].min(), raw[c].max(), raw[c].mean()
            inputs[c] = form.number_input(c, float(mn), float(mx), float(mean))
    submit = form.form_submit_button("🚀 Predict Manually")

        # ------------------------------
    # Manual Prediction
    # ------------------------------
    if submit:

        input_df = pd.DataFrame([inputs])

        st.write("🔍 Input Data:")
        st.dataframe(input_df)

        # Safety check
        if input_df.empty:
            st.error("Input data is empty. Please fill all fields.")
            st.stop()

        try:
            # Ensure correct column order
            if models.get("placement_pipe"):
                expected_cols = models["placement_pipe"].feature_names_in_
                input_df = input_df.reindex(columns=expected_cols, fill_value=0)

                with st.spinner("Predicting placement..."):
                    proba = models["placement_pipe"].predict_proba(input_df)[0][1]

                st.metric("Placement Probability", f"{proba*100:.2f}%")

                if proba >= 0.5:
                    st.success("🎉 Likely to be PLACED")
                else:
                    st.error("❌ Likely NOT placed")

            # Salary Prediction
            if models.get("salary_reg"):
                try:
                    sal = models["salary_reg"].predict(input_df)[0]
                    st.info(f"💰 Estimated Salary: ₹{int(sal):,}")
                except Exception as e:
                    st.warning(f"Salary prediction skipped: {e}")

            # Job Role Prediction
            if models.get("role_clf") and models.get("le_jobrole"):
                try:
                    role_enc = models["role_clf"].predict(input_df)[0]
                    role = models["le_jobrole"].inverse_transform([int(role_enc)])[0]
                    st.info(f"👔 Predicted Job Role: {role}")
                except Exception as e:
                    st.warning(f"Role prediction skipped: {e}")

        except Exception as e:
            st.error(f"Prediction Error: {e}")
# ==========================================================
# TAB 4: DOCS
# ==========================================================
with tab4:
    st.header("📘 Project Documentation & Artifacts")
    st.markdown("""
    ### 🔥 Features
    - Interactive EDA with KPIs, Pie, Box, Violin, Scatter, Heatmap
    - Multi-model training (RandomForest, Logistic, XGBoost optional)
    - Predict Placement, Salary, Job Role
    - Save & Load trained models (`artifacts/` folder)

    ### ⚙️ How to Run
    1. `pip install -r requirements.txt`
    2. (optional) `python generate_dataset.py`
    3. `python train_models.py`
    4. `streamlit run app.py`

    ### 📌 Next Steps
    - Add SHAP explainability for predictions
    - Deploy on Streamlit Cloud / Render
    - Student profile system with login
    """)

    st.subheader("Artifacts Folder")
    st.write(os.listdir(ARTIFACT_DIR))

import os
import json
import pickle
import numpy as np
import pandas as pd
import shap
import streamlit as st
import streamlit.components.v1 as components

# =========================================================
# 1. load web artifacts
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(WEB_DIR, "final_pipeline.pkl"), "rb") as f:
    final_pipeline = pickle.load(f)

with open(os.path.join(WEB_DIR, "model_meta.json"), "r", encoding="utf-8") as f:
    meta = json.load(f)

X_bg = pd.read_csv(os.path.join(WEB_DIR, "shap_background.csv"))

candidate_features = meta["candidate_features"]
optimal_threshold = float(meta["optimal_threshold"])
best_model_name = meta["best_model_name"]

# =========================================================
# 2. page settings
# =========================================================
st.set_page_config(
    page_title="Psoriasis-Associated Insulin Resistance Risk Prediction Tool",
    page_icon="🩺",
    layout="centered"
)

st.markdown("""
<style>
.main > div {
    max-width: 1100px;
    padding-top: 1.5rem;
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}
.stButton > button {
    background-color: #d9534f;
    color: white;
    border: 2px solid #d9534f;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.5rem 1.2rem;
}
.stButton > button:hover {
    background-color: white;
    color: #d9534f;
    border: 2px solid #d9534f;
}
.result-box {
    text-align: center;
    font-size: 2rem;
    font-weight: 700;
    margin-top: 1rem;
    margin-bottom: 1rem;
}
.result-note {
    text-align: center;
    font-size: 1.15rem;
    font-style: italic;
    margin-top: 0.8rem;
    margin-bottom: 1rem;
}
.small-note {
    color: #666;
    font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align:center; font-size: 2.2rem;'>Psoriasis-Associated Insulin Resistance Risk Prediction Tool</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;' class='small-note'>Enter the required clinical variables to obtain an individualized IR risk estimate and SHAP force plot explanation.</p>",
    unsafe_allow_html=True
)

# =========================================================
# 3. sidebar / form
# =========================================================
with st.form("prediction_form", clear_on_submit=False):
    age = st.number_input("Age", min_value=0.0, max_value=120.0, value=45.0, step=1.0)
    sex_label = st.selectbox("Sex", ["Female", "Male"])
    sex = 0 if sex_label == "Female" else 1
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.0, step=0.1)
    pasi = st.number_input("PASI", min_value=0.0, max_value=72.0, value=8.0, step=0.1)
    duration = st.number_input("Disease duration (years)", min_value=0.0, max_value=80.0, value=10.0, step=0.1)
    hdl = st.number_input("HDL", min_value=0.1, max_value=5.0, value=1.2, step=0.01)
    ldl = st.number_input("LDL", min_value=0.1, max_value=10.0, value=2.8, step=0.01)

    submitted = st.form_submit_button("Predict")

# =========================================================
# 4. prediction
# =========================================================
if submitted:
    input_df = pd.DataFrame([{
    "age": float(age),
    "sex": int(sex),
    "bmi": float(bmi),
    "pasi": float(pasi),
    "duration": float(duration),
    "hdl": float(hdl),
    "ldl": float(ldl),
    }])
    prob = float(final_pipeline.predict_proba(input_df)[:, 1][0])
    pred_class = int(prob >= optimal_threshold)

    risk_pct = prob * 100
    pred_text = "IR" if pred_class == 1 else "Non-IR"
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;'>Prediction Result</h2>", unsafe_allow_html=True)
    st.markdown(
    f"<div class='result-note'>Based on feature values, predicted probability of insulin resistance is <b>{risk_pct:.1f}%</b>.</div>",
    unsafe_allow_html=True
    )
    
    st.markdown(
    f"<div style='text-align:center; font-size:1.1rem; margin-bottom:0.8rem;'><b>Predicted class:</b> {pred_text}</div>",
    unsafe_allow_html=True
    )
    
    with st.expander("Show technical details"):
        st.write(f"**Final model:** {best_model_name}")
        st.write(f"**Predicted probability of IR:** {prob:.3f}")
        st.write(f"**Locked threshold:** {optimal_threshold:.3f}")

    # =====================================================
    # 5. SHAP force plot
    # =====================================================
    preprocess = final_pipeline.named_steps["preprocess"]
    model = final_pipeline.named_steps["model"]

    X_bg_t = preprocess.transform(X_bg)
    X_input_t = preprocess.transform(input_df)

    if hasattr(X_bg_t, "toarray"):
        X_bg_t = X_bg_t.toarray()
    if hasattr(X_input_t, "toarray"):
        X_input_t = X_input_t.toarray()

    X_bg_t = np.asarray(X_bg_t)
    X_input_t = np.asarray(X_input_t)

    try:
        feature_names = preprocess.get_feature_names_out().tolist()
        feature_names = [str(x).replace("num__", "").replace("cat__", "") for x in feature_names]
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_input_t.shape[1])]

    tree_model_names = [
        "RandomForestClassifier", "ExtraTreesClassifier",
        "GradientBoostingClassifier", "XGBClassifier",
        "LGBMClassifier", "CatBoostClassifier",
        "DecisionTreeClassifier"
    ]
    model_name = model.__class__.__name__

    if model_name in tree_model_names or hasattr(model, "feature_importances_"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_input_t)

        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1] if np.ndim(base_value) > 0 else float(base_value)
        base_value = float(np.asarray(base_value))

    else:
        bg_small = X_bg_t[:min(50, len(X_bg_t))]

        def pred_fn(x):
            return model.predict_proba(x)[:, 1]

        explainer = shap.KernelExplainer(pred_fn, bg_small)
        shap_values = explainer.shap_values(X_input_t, nsamples="auto")

        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1] if np.ndim(base_value) > 0 else float(base_value)
        base_value = float(np.asarray(base_value))

    shap.initjs()
    force = shap.force_plot(
        base_value=base_value,
        shap_values=shap_values[0],
        features=X_input_t[0],
        feature_names=feature_names,
        matplotlib=False
    )

    temp_html = os.path.join(WEB_DIR, "temp_force_plot.html")
    shap.save_html(temp_html, force)

    with open(temp_html, "r", encoding="utf-8") as f:
        html_string = f.read()

    st.markdown(
    "<h3 style='text-align:center; margin-top:1rem;'>Individualized SHAP Force Plot</h3>",
    unsafe_allow_html=True
    )
    components.html(html_string, height=260, scrolling=False)

    # =====================================================
    # 6. show raw inputs
    # =====================================================
    st.subheader("Input Summary")
    st.dataframe(input_df, use_container_width=True)

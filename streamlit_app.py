import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix, classification_report

st.set_page_type = "wide"
st.title("💤 Sleep Debt Risk Classification Dashboard")
st.write("Upload your test dataset to evaluate and compare 5 distinct Machine Learning classifiers.")

# Sidebar Configuration
st.sidebar.header("Configuration Panel")
uploaded_file = st.sidebar.file_uploader("Upload Test CSV (test_data.csv)", type=["csv"])

model_choice = st.sidebar.selectbox(
    "Choose ML Classifier",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Support Vector Machine"]
)

# Load pipeline dependencies safely
@st.cache_resource
def load_artifacts():
    scaler = joblib.load('model/scaler.joblib')
    target_le = joblib.load('model/target_encoder.joblib')
    return scaler, target_le

try:
    scaler, target_le = load_artifacts()
except Exception as e:
    st.error("Model artifacts missing. Please ensure the 'model/' directory is populated inside your repository.")
    st.stop()

if uploaded_file is not None:
    # Process test data without retraining
    test_df = pd.read_csv(uploaded_file)
    
    if 'sleep_debt_category' not in test_df.columns:
        st.error("The uploaded file must contain the target label column: 'sleep_debt_category'")
    else:
        # Separate features and target
        X_test = test_df.drop(columns=['sleep_debt_category'])
        y_test = test_df['sleep_debt_category']
        
        # Strip user_id if present
        if 'user_id' in X_test.columns:
            X_test = X_test.drop(columns=['user_id'])

        leakage_columns = ['next_day_fatigue_score', 'total_sleep_hours']
        for col in leakage_columns:
            if col in X_test.columns:
                X_test = X_test.drop(columns=[col])            
            
        # Transform structural numeric representations
        X_test_scaled = scaler.transform(X_test)
        
        # Load Selected Model dynamically
        model_filename = f"model/{model_choice.lower().replace(' ', '_')}.joblib"
        model = joblib.load(model_filename)
        
        # Inference Generation
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)
        
        # Compute core metrics
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='macro', zero_division=0)
        rec = recall_score(y_test, preds, average='macro')
        f1 = f1_score(y_test, preds, average='macro')
        mcc = matthews_corrcoef(y_test, preds)
        auc = roc_auc_score(y_test, probs, average='macro', multi_class='ovr')
        
        # Layout metrics layout visually
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy Score", f"{acc:.4f}")
        col2.metric("AUC Metric", f"{auc:.4f}")
        col3.metric("F1-Score (Macro)", f"{f1:.4f}")
        
        col4, col5, col6 = st.columns(3)
        col4.metric("Precision", f"{prec:.4f}")
        col5.metric("Recall Rate", f"{rec:.4f}")
        col6.metric("MCC Score", f"{mcc:.4f}")
        
        # Diagnostics Report Plots
        st.subheader(f"Performance Architecture: {model_choice}")
        
        tab1, tab2 = st.tabs(["Confusion Matrix Map", "Classification Report Text"])
        
        with tab1:
            cm = confusion_matrix(y_test, preds)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=target_le.classes_, yticklabels=target_le.classes_, ax=ax)
            plt.ylabel('Actual Category')
            plt.xlabel('Predicted Category')
            st.pyplot(fig)
            
        with tab2:
            report = classification_report(y_test, preds, target_names=target_le.classes_, output_dict=False)
            st.code(report)
else:
    st.info("💡 Please upload the generated 'test_data.csv' via the sidebar profile panel to begin analysis.")
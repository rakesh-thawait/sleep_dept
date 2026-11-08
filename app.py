import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score

# 1. Load data (Assuming your data is saved as bedtime_screentime_sleep_debt.csv)
df = pd.read_csv('bedtime_screentime_sleep_debt.csv')

# Drop identifier
df = df.drop(columns=['user_id'])

# 2. Encode Categorical Features
categorical_cols = ['gender', 'occupation_type', 'chronotype', 'primary_bedtime_app']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Encode Target
target_le = LabelEncoder()
df['sleep_debt_category'] = target_le.fit_transform(df['sleep_debt_category'])

# 3. Features and Target Split
# Drop target proxies to eliminate deterministic data leakage
leakage_columns = ['next_day_fatigue_score', 'total_sleep_hours']

X = df.drop(columns=['sleep_debt_category'] + leakage_columns)
y = df['sleep_debt_category']

# Split train/test (saving test data for your Streamlit app upload)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save the test data to a CSV as requested by the assignment
test_df = X_test.copy()
test_df['sleep_debt_category'] = y_test
test_df.to_csv('test_data.csv', index=False)

# 4. Scale Numeric Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create model directory
os.makedirs('model', exist_ok=True)
joblib.dump(scaler, 'model/scaler.joblib')
joblib.dump(label_encoders, 'model/label_encoders.joblib')
joblib.dump(target_le, 'model/target_encoder.joblib')

# 5. Initialize Models
models = {
    'Logistic Regression': OneVsRestClassifier(LogisticRegression(max_iter=1000)),
    'Decision Tree': DecisionTreeClassifier(
        max_depth=5,              # Restricts the tree from growing too deep
        min_samples_split=10,     # Requires at least 10 samples to split a node
        min_samples_leaf=5,       # Requires at least 5 samples in a leaf node
        random_state=42
    ),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Support Vector Machine': SVC(probability=True, random_state=42)
}

# 6. Train and Evaluate
results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)
    
    # Calculate metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='macro', zero_division=0)
    rec = recall_score(y_test, preds, average='macro')
    f1 = f1_score(y_test, preds, average='macro')
    mcc = matthews_corrcoef(y_test, preds)
    auc = roc_auc_score(y_test, probs, average='macro', multi_class='ovr')
    
    results.append({
        'ML Model Name': name,
        'Accuracy': round(acc, 4),
        'AUC': round(auc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1': round(f1, 4),
        'MCC': round(mcc, 4)
    })
    
    # Save trained model artifact
    joblib.dump(model, f'model/{name.lower().replace(" ", "_")}.joblib')

# Print out evaluation summary matrix for your README
print(pd.DataFrame(results).to_markdown(index=False))

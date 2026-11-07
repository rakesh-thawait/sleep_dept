# M.Tech AIML - Machine Learning Assignment 2

## a. Problem Statement
The baseline objective of this project is to construct, evaluate, and deploy an end-to-end classification pipeline that predicts an individual's chronic behavioral sleep state (`sleep_debt_category`). Using digital signature points tracking night-time screen interaction habits, demographic information, and biochemical drivers (caffeine, exercise latency), the pipeline contrasts 5 structural classification paradigms to evaluate structural diagnostic health risks.

## b. Dataset Description
The underlying dataset features 1,000 empirical observations mapping lifestyle metrics across 18 unique structural parameters:
- **Demographics:** Age, Gender, Occupation Type (Categorical fields tracking shift workloads).
- **Behavioral Variables:** Bedtime Phone Minutes, Primary Bedtime App, Screen Brightness Pct, Blue Light Filter Active, Caffeine Post 5 PM (mg), Physical Activity Min.
- **Biometric Targets:** Sleep Latency Min, Total Sleep Hours, Deep Sleep Pct, REM Sleep Pct, Morning Alarm Snoozes, Next Day Fatigue Score.
- **Target Label (`sleep_debt_category`):** Multi-class category mapping objective debt risk profiles ('Severe Sleep Debt', 'Moderate Debt', 'Mild Deficit', 'Optimal Recovery').

## c. GitHub Repository Link
[Insert Your Clickable GitHub Repository URL Here]

## d. Models Used & Evaluation Comparison Matrix

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.9035 | 0.9852 | 0.9373 | 0.8686 | 0.8958 | 0.8498 |
| Decision Tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| KNN | 0.7912 | 0.9285 | 0.7833 | 0.7271 | 0.7500 | 0.6680 |
| Naive Bayes | 0.8782 | 0.9839 | 0.8444 | 0.9018 | 0.8686 | 0.8156 |
| Support Vector Machine | 0.9541 | 0.9970 | 0.9447 | 0.9411 | 0.9428 | 0.9279 |

### Observations on Performance Paradigms

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieves 0.9035 accuracy and 0.9852 AUC, with macro precision (0.9373) higher than macro recall (0.8686). |
| **Decision Tree** | Scores 1.0000 across all reported metrics on this test split. Validate this perfect result on additional splits or with cross-validation to check for overfitting or data leakage. |
| **KNN** | Records the lowest accuracy (0.7912) and MCC (0.6680) among the evaluated models, while its AUC is 0.9285. |
| **Naive Bayes** | Reaches 0.8782 accuracy and 0.9839 AUC, with recall (0.9018) exceeding precision (0.8444). |
| **Support Vector Machine** | Is the strongest model after the Decision Tree, with 0.9541 accuracy, 0.9970 AUC, and 0.9279 MCC. |

### 🏆 Overall Dataset Winner
**Decision Tree** is the top performer on this evaluation split, achieving **1.0000** for accuracy, AUC, precision, recall, F1, and MCC. Because perfect test metrics can indicate overfitting or data leakage, validate this result with additional splits or cross-validation before treating it as generalizable.

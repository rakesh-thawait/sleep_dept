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
| Logistic Regression | 0.8450 | 0.9420 | 0.8310 | 0.8290 | 0.8300 | 0.7850 |
| Decision Tree | 0.8900 | 0.9120 | 0.8840 | 0.8910 | 0.8870 | 0.8420 |
| KNN Classifier | 0.8150 | 0.9010 | 0.8020 | 0.7980 | 0.8000 | 0.7310 |
| Naive Bayes | 0.7600 | 0.8850 | 0.7520 | 0.7490 | 0.7500 | 0.6620 |
| Support Vector Machine | 0.9150 | 0.9680 | 0.9110 | 0.9130 | 0.9120 | 0.8810 |

### Observations on Performance Paradigms

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Shows steady structural performance with high AUC linear separations, though minor baseline degradation occurs due to multi-class non-linear interaction spaces. |
| **Decision Tree** | Captures high variance and contextual boundaries effectively; maps out specific split points cleanly across parameters like brightness and phone active time. |
| **KNN Classifier** | Solid local non-parametric clustering metrics, though structural neighborhood performance drops slightly due to multi-dimensional scale density variants. |
| **Naive Bayes** | Executes fast baseline multi-class processing, but its assumption of variable independence limits precision scores across closely coupled sleep metrics. |
| **Support Vector Machine** | Achieves top performance by using optimal soft-margin boundaries via an RBF space kernel to handle complex non-linear overlaps. |

### 🏆 Overall Dataset Winner
**Support Vector Machine (SVM)** stands out as the winner for this dataset, achieving peak performance across all baseline criteria (**Accuracy: 91.50%**, **MCC Score: 0.8810**).

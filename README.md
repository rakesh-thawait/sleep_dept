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
https://github.com/rakesh-thawait/sleep_dept

## d. Models Used & Evaluation Comparison Matrix

| ML Model Name          |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:-----------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression    |     0.6947 | 0.8897 |      0.7058 |   0.5976 | 0.6271 | 0.4993 |
| Decision Tree          |     0.7365 | 0.9097 |      0.7235 |   0.7145 | 0.7169 | 0.5942 |
| KNN                    |     0.6465 | 0.8378 |      0.6441 |   0.5564 | 0.5869 | 0.4305 |
| Naive Bayes            |     0.6824 | 0.8955 |      0.6340 |   0.6824 | 0.6532 | 0.5181 |
| Support Vector Machine |     0.7441 | 0.9166 |      0.7336 |   0.6920 | 0.7106 | 0.5911 |

### Observations on Performance Paradigms

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Demonstrates solid, balanced multi-class performance with an AUC of 0.8897, confirming stable probabilistic separation despite the high-dimensional non-linear interactions within bedtime behaviors. |
| **Decision Tree** | With structural hyperparameter tuning (max_depth=5) and target proxies removed, the model generalizes realistically at 73.65% accuracy, successfully tracking key split-rules like screen brightness and minutes active. |
| **KNN** | Yields the lowest performance metrics (64.65% Accuracy, 0.4305 MCC), showing that instance-based distance metrics struggle slightly within the scaled variance of combined categorical and numerical sleep profiles. |
| **Naive Bayes** | Functions effectively with a high multi-class AUC of 0.8955. It exhibits a stronger Recall (0.6824) than Precision (0.6340), indicating a tendency to flag risk categories aggressively despite feature correlation. |
| **Support Vector Machine** | Achieves peak classification boundary separation across the dataset. The RBF kernel handles non-linear overlaps smoothly, yielding top-tier metrics across the board (0.7441 Accuracy, 0.9166 AUC). |

### 🏆 Overall Winner for your dataset?
The **Support Vector Machine (SVM)** is the definitive winner for this dataset. It achieves the highest **Accuracy (74.41%)** and **AUC (0.9166)**, demonstrating robust structural generalization capabilities across multi-class risk categories without over-relying on deterministic target leakage.

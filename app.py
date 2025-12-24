import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

data = pd.read_csv("heart.csv")

X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

logistic_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

rf_pipeline = Pipeline([
    ("model", RandomForestClassifier(random_state=42))
])

logistic_params = {
    "model__C": [0.01, 0.1, 1, 10],
    "model__solver": ["liblinear"]
}

rf_params = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5, 10],
    "model__min_samples_split": [2, 5]
}

logistic_grid = GridSearchCV(
    logistic_pipeline,
    logistic_params,
    cv=5,
    scoring="roc_auc"
)

rf_grid = GridSearchCV(
    rf_pipeline,
    rf_params,
    cv=5,
    scoring="roc_auc"
)

logistic_grid.fit(X_train, y_train)
rf_grid.fit(X_train, y_train)

best_logistic = logistic_grid.best_estimator_
best_rf = rf_grid.best_estimator_

logistic_pred = best_logistic.predict(X_test)
rf_pred = best_rf.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, logistic_pred))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

print("\nLogistic Regression Report:\n", classification_report(y_test, logistic_pred))
print("\nRandom Forest Report:\n", classification_report(y_test, rf_pred))

logistic_probs = best_logistic.predict_proba(X_test)[:,1]
rf_probs = best_rf.predict_proba(X_test)[:,1]

fpr_l, tpr_l, _ = roc_curve(y_test, logistic_probs)
fpr_r, tpr_r, _ = roc_curve(y_test, rf_probs)

auc_l = auc(fpr_l, tpr_l)
auc_r = auc(fpr_r, tpr_r)

plt.figure()
plt.plot(fpr_l, tpr_l, label=f"Logistic Regression AUC = {auc_l:.2f}")
plt.plot(fpr_r, tpr_r, label=f"Random Forest AUC = {auc_r:.2f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

def predict_heart_disease(input_data):
    input_df = pd.DataFrame([input_data], columns=X.columns)
    prediction = best_rf.predict(input_df)
    probability = best_rf.predict_proba(input_df)[0][1]
    return prediction[0], probability

sample_patient = {
    "age": 54,
    "sex": 1,
    "cp": 0,
    "trestbps": 130,
    "chol": 246,
    "fbs": 0,
    "restecg": 1,
    "thalach": 173,
    "exang": 0,
    "oldpeak": 0.0,
    "slope": 1,
    "ca": 0,
    "thal": 2
}

result, prob = predict_heart_disease(sample_patient)

if result == 1:
    print(f"Heart Disease Detected (Risk Probability: {prob:.2f})")
else:
    print(f"No Heart Disease (Risk Probability: {prob:.2f})")

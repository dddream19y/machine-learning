import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score   #模型選取與交叉驗證
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2   #特徵選取
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib

df = pd.read_csv("heart_disease.csv")

df.head()
df.info()
df.shape
df.isnull().sum()
df.describe()

print(df['target'].value_counts())

X = df.drop('target', axis=1)
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, "scaler.pkl")

#相關係數圖
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# 資料分佈圖
sns.countplot(x='target', data=df)
plt.title("Target Variable Distribution")
plt.show()

# 單變量分析(年齡、血壓、膽固醇、最大心率、oldpeak)
features_to_plot = ['age', 'resting bp s', 'cholesterol', 'max heart rate', 'oldpeak']
for feature in features_to_plot:
    plt.figure(figsize=(6,4))
    sns.histplot(df[feature], kde=True, bins=30)
    plt.title(f"Distribution of {feature}")
    plt.show()

# 雙變量分析(年齡、血壓、膽固醇、最大心率、oldpeak)
for feature in features_to_plot:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='target', y=feature, data=df)
    plt.title(f"Boxplot of {feature} by Target")
    plt.show()

# Feature Selection
selector = SelectKBest(score_func=chi2, k=8)  #選取前8個特徵
X_selected = selector.fit_transform(np.abs(X_scaled), y)

selected_features = selector.get_support(indices=True)
print("Selected Feature Indices:", selected_features)
print("Selected Feature Names:", X.columns[selected_features].tolist())

# Save the selector
joblib.dump(selector, "selector.pkl")

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

#Logistic Regression Model
# Defining Hyperparameter Grid
param_grid_log = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs']
}

# Setting up GridSearchCV
grid_log = GridSearchCV(LogisticRegression(max_iter=1000), param_grid_log, cv=5, scoring='accuracy', n_jobs=-1)

# Fitting the model
grid_log.fit(X_train, y_train)

# Best Estimator
best_log = grid_log.best_estimator_

print("Best Hyperparameters for Logistic Regression:", grid_log.best_params_)
#{'C': 100, 'solver': 'lbfgs'}

# Prediction
y_pred_log = best_log.predict(X_test)
y_prob_log = best_log.predict_proba(X_test)[:, 1]

# Evaluation
print("Classification Report for Logistic Regression:")
print(classification_report(y_test, y_pred_log))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_log)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

print("ROC AUC Score:", roc_auc_score(y_test, y_prob_log))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob_log)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc='lower right')
plt.show()

#Random Forest Model
# Defining Hyperparameter Grid
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8, 10],
    'min_samples_split': [2, 5, 10]
}

grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)

grid_rf.fit(X_train, y_train)

best_rf = grid_rf.best_estimator_

print("Best Hyperparameters for Random Forest:", grid_rf.best_params_)

# Prediction
y_pred_rf = best_rf.predict(X_test)
y_prob_rf = best_rf.predict_proba(X_test)[:, 1]

print("Classification Report for Random Forest:")
print(classification_report(y_test, y_pred_rf))

cm_rf = confusion_matrix(y_test, y_pred_rf)

# Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Random Forest')
plt.show()
print("ROC AUC Score:", roc_auc_score(y_test, y_prob_rf))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob_rf)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend(loc='lower right')
plt.show()


# Model Comparison
model_results = {
    "Logistic Regression": [accuracy_score(y_test, y_pred_log), roc_auc_score(y_test, y_prob_log)],
    "Random Forest": [accuracy_score(y_test, y_pred_rf), roc_auc_score(y_test, y_prob_rf)]
}

results_df = pd.DataFrame(model_results, index=["Accuracy", "ROC AUC"]).T
print(results_df)

# Bar plot for Accuracy and ROC AUC
results_df.plot(kind='bar', figsize=(10,6))
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.ylim(0.5, 1.0)
plt.xticks(rotation=45)
plt.legend(loc='lower right')
plt.show()

import joblib

# Save the best Random Forest model
# joblib.dump(results_df, "final_heart_disease_model.pkl")
joblib.dump(best_log, "final_heart_disease_model.pkl")
print("Random Forest Model saved successfully!")

def predict_heart_disease(input_data):
    """
    Predict heart disease from new patient data.
    input_data: List of original 11 feature values
    """
    model = joblib.load("final_heart_disease_model.pkl")
    scaler = joblib.load("scaler.pkl")
    selector = joblib.load("selector.pkl")

    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    input_selected = selector.transform(np.abs(input_scaled))

    prediction = model.predict(input_selected)
    probability = model.predict_proba(input_selected)[0][1]

    if prediction[0] == 1:
        print(f"Prediction: Positive for Heart Disease with probability {probability:.2f}")
    else:
        print(f"Prediction: Negative for Heart Disease with probability {1 - probability:.2f}")

predict_heart_disease([63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 1])

predict_heart_disease([58, 1, 2, 150, 270, 0, 1, 111, 1, 2.5, 2])


predict_heart_disease([49,0,3,160,180,0,0,156,0,1,2])

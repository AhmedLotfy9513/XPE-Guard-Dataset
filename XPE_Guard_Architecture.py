import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the balanced dataset
df = pd.read_csv("XPE_Guard_Dataset.csv")

# Separate features and labels
X = df.drop(columns=['label', 'Family_Label'], errors='ignore')
y = df['label']

# Split data 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Train XGBoost model
model = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
model.fit(X_train, y_train)

# Evaluate model
preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print("\nClassification Report:")
print(classification_report(y_test, preds))

# Generate SHAP plot for explainability
explainer = shap.TreeExplainer(model)
X_sample = X_train.sample(n=1000, random_state=42)
shap_values = explainer.shap_values(X_sample)

plt.figure()
shap.summary_plot(shap_values, X_sample, show=False)
plt.tight_layout()
plt.savefig("shap_summary.jpg", dpi=300)
print("SHAP summary plot saved as shap_summary.jpg")

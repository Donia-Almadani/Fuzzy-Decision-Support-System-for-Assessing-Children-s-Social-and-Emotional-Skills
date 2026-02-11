import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# ==============================
# ==============================
def classify_level(score):
    if score < 4:
        return 'Needs Support'
    elif score < 7:
        return 'Developing'
    else:
        return 'Strong'

# ==============================
# Load data
# ==============================
data = pd.read_excel("fuzzy_results_with_feedback.xlsx")


data = data.dropna(subset=['social_skill_result'])

# Select feature
X = data[['empathy', 'cooperation', 'emotional_reg', 'school_Belonging']]
y = data['social_skill_result']

# ==============================
# Split data (train\ test)
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
#  Random Forest
# ==============================
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ==============================
# (Regression)
# ==============================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMachine Learning Validation Results")
print("--------------------------------------")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# ==============================
#  (Classes)
# ==============================
# real value
y_test_levels = y_test.apply(classify_level)

# predicted value
y_pred_series = pd.Series(y_pred, index=y_test.index)
y_pred_levels = y_pred_series.apply(classify_level)

levels_order = ['Needs Support', 'Developing', 'Strong']

# ==============================
# 6) Confusion Matrix + Report
# ==============================
cm = confusion_matrix(y_test_levels, y_pred_levels, labels=levels_order)
print("\nConfusion Matrix (rows = true, cols = predicted):")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test_levels, y_pred_levels, target_names=levels_order))

# confusion matrix graph
plt.figure(figsize=(5, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=levels_order)
disp.plot(values_format='d', cmap="Blues")
plt.title("Confusion Matrix for Social Skill Levels")
plt.tight_layout()
plt.show()

# ==============================
#  Feature Importance Plot
# ==============================
importance = model.feature_importances_
plt.figure(figsize=(7, 5))
plt.bar(X.columns, importance)
plt.title("Feature Importance in Predicting Social Skill Result")
plt.ylabel("Importance Score")
plt.xlabel("Feature")
plt.tight_layout()
plt.show()

# ==============================
#  Excel
# ==============================
pred_df = X_test.copy()
pred_df["actual_result"] = y_test.values
pred_df["predicted_result"] = y_pred
pred_df["actual_level"] = y_test_levels.values
pred_df["predicted_level"] = y_pred_levels.values

pred_df.to_excel("ml_validation_results.xlsx", index=False)

print("Saved ML validation results (with levels) to ml_validation_results.xlsx\n")
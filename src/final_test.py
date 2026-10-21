
import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from preprocessing import (
    x_train, x_val, y_train, y_val,
    x_test, y_test, preprocessor
)

# Combine training and validation data after model selection
X_final_train = pd.concat([x_train, x_val], axis=0)
y_final_train = pd.concat([y_train, y_val], axis=0)

# Build the selected MLP model
final_model = Pipeline([
    ("preprocess", preprocessor),
    ("classifier", MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.0003,
        alpha=0.001,
        batch_size=64,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25,
        random_state=42,
    )),
])

print("\nTraining final MLP model...")
final_model.fit(X_final_train, y_final_train)

# Evaluate on the held-out test set
test_predictions = final_model.predict(x_test)
test_probabilities = final_model.predict_proba(x_test)[:, 1]

accuracy = accuracy_score(y_test, test_predictions)
precision = precision_score(
    y_test, test_predictions, zero_division=0
)
recall = recall_score(
    y_test, test_predictions, zero_division=0
)
f1 = f1_score(y_test, test_predictions, zero_division=0)
roc_auc = roc_auc_score(y_test, test_probabilities)

print("\n========== FINAL TEST RESULTS ==========")
print(f"Test Accuracy : {accuracy * 100:.2f}%")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall   : {recall:.4f}")
print(f"Test F1 Score : {f1:.4f}")
print(f"Test ROC-AUC  : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_predictions))

print("\nClassification Report:")
print(classification_report(
    y_test,
    test_predictions,
    target_names=["No Churn", "Churn"],
    zero_division=0,
))

# Save the complete pipeline
os.makedirs("models", exist_ok=True)
joblib.dump(final_model, "models/churn_mlp_model.joblib")

print("\nFinal model saved to models/churn_mlp_model.joblib")
print("ChurnSense AI final evaluation completed.")

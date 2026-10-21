import os
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
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
    x_train,
    x_test,
    y_train,
    y_test,
    preprocessor,
)
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42,
    )),
])
print("Training final MLP model...")
model.fit(x_train, y_train)
probabilities = model.predict_proba(x_test)[:, 1]
threshold = 0.45
predictions = (probabilities >= threshold).astype(int)
metrics = {
    "Threshold": threshold,
    "Accuracy": accuracy_score(y_test, predictions),
    "Precision": precision_score(
        y_test, predictions, zero_division=0
    ),
    "Recall": recall_score(
        y_test, predictions, zero_division=0
    ),
    "F1 Score": f1_score(
        y_test, predictions, zero_division=0
    ),
    "ROC AUC": roc_auc_score(y_test, probabilities),
}

print("\nFinal Results:")
for name, value in metrics.items():
    if name == "Threshold":
        print(f"{name}: {value:.2f}")
    else:
        print(f"{name}: {value:.4f} ({value * 100:.2f}%)")
cm = confusion_matrix(y_test, predictions)
print("\nConfusion Matrix:")
print(cm)
print("\nRows = actual classes; columns = predicted classes")
print("Class 0 = No Churn; Class 1 = Churn")
print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["No Churn", "Churn"],
    zero_division=0,
))
os.makedirs("reports", exist_ok=True)
pd.DataFrame([metrics]).to_csv(
    "reports/final_test_metrics.csv",
    index=False,
)
print("Metrics saved to reports/final_test_metrics.csv")

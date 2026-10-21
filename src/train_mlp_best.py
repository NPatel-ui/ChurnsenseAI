
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from preprocessing import (
    x_train, x_val, y_train, y_val, preprocessor
)

model = Pipeline([
    ("preprocess", preprocessor),
    ("classifier", MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        learning_rate_init=0.0005,
        alpha=0.001,
        batch_size=64,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25,
        random_state=42,
    )),
])

print("Training MLP Classifier...")
model.fit(x_train, y_train)

for name, X, y in [
    ("TRAINING", x_train, y_train),
    ("VALIDATION", x_val, y_val),
]:
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    print(f"\n{name} RESULTS")
    print(f"Accuracy : {accuracy_score(y, predictions):.4f}")
    print(f"Precision: {precision_score(y, predictions, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y, predictions, zero_division=0):.4f}")
    print(f"F1 Score : {f1_score(y, predictions, zero_division=0):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y, probabilities):.4f}")

print("\nTraining iterations:",
      model.named_steps["classifier"].n_iter_)

print("\nMLP training completed.")

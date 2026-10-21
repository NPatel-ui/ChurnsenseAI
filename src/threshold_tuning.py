import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from preprocessing import (
    x_train,
    x_val,
    y_train,
    y_val,
    preprocessor,
)
mlp_model = Pipeline([
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
print("Training selected MLP model")
mlp_model.fit(x_train, y_train)
churn_probabilities = mlp_model.predict_proba(x_val)[:, 1]
thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
results = []
for threshold in thresholds:
    predictions = (churn_probabilities >= threshold).astype(int)
    results.append({
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_val, predictions),
        "Precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_val, predictions, zero_division=0
        ),
    })
results_df = pd.DataFrame(results)
print("\nThreshold Tuning Results:")
print(results_df.round(4).to_string(index=False))
results_df.to_csv(
    "reports/threshold_tuning_results.csv",
    index=False
)
print("\nResults saved to reports/threshold_tuning_results.csv")
print("\nROC-AUC:", round(
    roc_auc_score(y_val, churn_probabilities), 4
))
print("Average Precision:", round(
    average_precision_score(y_val, churn_probabilities), 4
))

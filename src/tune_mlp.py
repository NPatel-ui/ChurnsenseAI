
import os
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    average_precision_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from preprocessing import (
    x_train,
    x_val,
    y_train,
    y_val,
    preprocessor,
)


# Five controlled MLP experiments


experiments = [
    {
        "name": "Baseline",
        "layers": (128, 64, 32),
        "alpha": 0.001,
        "lr": 0.0003,
        "batch_size": "auto",
    },
    {
        "name": "Lower_Regularization",
        "layers": (128, 64, 32),
        "alpha": 0.0001,
        "lr": 0.0003,
        "batch_size": "auto",
    },
    {
        "name": "Higher_Regularization",
        "layers": (128, 64, 32),
        "alpha": 0.01,
        "lr": 0.0003,
        "batch_size": "auto",
    },
    {
        "name": "Batch_Size_32",
        "layers": (128, 64, 32),
        "alpha": 0.001,
        "lr": 0.0003,
        "batch_size": 32,
    },
    {
        "name": "Batch_Size_64",
        "layers": (128, 64, 32),
        "alpha": 0.001,
        "lr": 0.0003,
        "batch_size": 64,
    },
    {
        "name": "Batch_Size_128",
        "layers": (128, 64, 32),
        "alpha": 0.001,
        "lr": 0.0003,
        "batch_size": 128,
    },
]




results = []

for exp in experiments:
    print(f"\nTraining: {exp['name']}")

    model = Pipeline([
        ("preprocess", preprocessor),
        ("model", MLPClassifier(
    hidden_layer_sizes=exp["layers"],
    activation="relu",
    solver="adam",
    alpha=exp["alpha"],
    learning_rate_init=exp["lr"],
    batch_size=exp["batch_size"],
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,
    random_state=42,
)),

    ])

    model.fit(x_train, y_train)

    # Evaluate using the default threshold of 0.50
    predictions = model.predict(x_val)
    probabilities = model.predict_proba(x_val)[:, 1]

    network = model.named_steps["model"]

    results.append({
        "Experiment": exp["name"],
        "Accuracy": accuracy_score(y_val, predictions),
        "F1 Score": f1_score(
            y_val, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "Precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "Average Precision": average_precision_score(
            y_val, probabilities
        ),
        "ROC AUC": roc_auc_score(y_val, probabilities),
        "Iterations": network.n_iter_,
        "Final Loss": network.loss_,
        "Best Internal Validation Score": network.best_validation_score_,
    })


# Display and save comparison results
results_df = pd.DataFrame(results)

# Rank primarily by validation accuracy
results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\n========== MLP EXPERIMENT RESULTS ==========")
print(results_df.round(4).to_string(index=False))

os.makedirs("reports", exist_ok=True)

results_df.to_csv(
    "reports/mlp_results.csv",
    index=False,
)

print("\nResults saved to reports/mlp_results.csv")

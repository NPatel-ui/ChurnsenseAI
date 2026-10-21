
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from preprocessing import (
    x_train, x_val, y_train, y_val, preprocessor
)

experiments = [
    {
        "name": "MLP_64_32",
        "layers": (64, 32),
        "learning_rate": 0.001,
        "alpha": 0.001,
    },
    {
        "name": "MLP_128_64_32",
        "layers": (128, 64, 32),
        "learning_rate": 0.001,
        "alpha": 0.001,
    },
    {
        "name": "MLP_128_64_32_LR_0003",
        "layers": (128, 64, 32),
        "learning_rate": 0.0003,
        "alpha": 0.001,
    },
    {
        "name": "MLP_128_64_32_Alpha_001",
        "layers": (128, 64, 32),
        "learning_rate": 0.0003,
        "alpha": 0.01,
    },
    {
        "name": "MLP_256_128_64",
        "layers": (256, 128, 64),
        "learning_rate": 0.0003,
        "alpha": 0.001,
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
            learning_rate_init=exp["learning_rate"],
            alpha=exp["alpha"],
            batch_size=64,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42,
        )),
    ])

    model.fit(x_train, y_train)

    predictions = model.predict(x_val)
    probabilities = model.predict_proba(x_val)[:, 1]

    results.append({
        "Experiment": exp["name"],
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
        "ROC AUC": roc_auc_score(y_val, probabilities),
        "Average Precision": average_precision_score(
            y_val, probabilities
        ),
        "Iterations": model.named_steps["model"].n_iter_,
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="Accuracy", ascending=False
)

print("\n===== MLP V2 RESULTS =====")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    "reports/mlp_v2_results.csv",
    index=False
)

print("\nResults saved to reports/mlp_v2_results.csv")

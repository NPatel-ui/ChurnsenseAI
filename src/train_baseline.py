from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, roc_auc_score
from preprocessing import x_train, x_val, y_train, y_val,preprocessor
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000,random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
}
results=[]
for name, estimator in models.items():
    pipeline=Pipeline([
        ("preprocess",preprocessor),
        ("classifier", estimator)
    ])
    print(f"Training {name}")
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_val)
    probabilities = pipeline.predict_proba(x_val)[:, 1]
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_val, predictions),
        "Precision": precision_score(y_val, predictions, zero_division=0),
        "Recall": recall_score(y_val, predictions, zero_division=0),
        "F1": f1_score(y_val, predictions, zero_division=0),
        "Average Precision": average_precision_score(y_val, probabilities),
        "ROC AUC": roc_auc_score(y_val, probabilities)
    }
    results.append(metrics)
    print("Validation Metrics:")
    for metric, value in metrics.items():
        if metric != "Model":
            print(f"{metric}: {value:.4f}")
print("Baseline Comparison:")
print(__import__("pandas").DataFrame(results).to_string(index=False))
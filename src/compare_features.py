
import pandas as pd

from sklearn.base import clone
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
    x_train,
    x_val,
    y_train,
    y_val,
    preprocessor,
)


# Original feature columns from the existing split
original_features = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

engineered_features = [
    "AvgMonthlySpend",
    "ServicesCount",
    "TenureGroup",
]

feature_sets = {
    "Original_Features": original_features,
    "Original_Plus_AvgSpend": original_features + ["AvgMonthlySpend"],
    "Original_Plus_ServicesCount": original_features + ["ServicesCount"],
    "Original_Plus_TenureGroup": original_features + ["TenureGroup"],
    "All_Engineered_Features": original_features + engineered_features,
}


results = []

for name, columns in feature_sets.items():
    print(f"Training: {name}")

    # Select only the features for this experiment
    X_train_exp = x_train[columns]
    X_val_exp = x_val[columns]

    # Rebuild preprocessing for the selected columns
    numeric_features = [
        col for col in columns
        if col in [
            "SeniorCitizen",
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "AvgMonthlySpend",
            "ServicesCount",
        ]
    ]

    categorical_features = [
        col for col in columns
        if col not in numeric_features
    ]

    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    experiment_preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    model = Pipeline([
        ("preprocess", experiment_preprocessor),
        ("model", MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.0003,
            batch_size=64,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42,
        )),
    ])

    model.fit(X_train_exp, y_train)

    predictions = model.predict(X_val_exp)
    probabilities = model.predict_proba(X_val_exp)[:, 1]

    results.append({
        "Experiment": name,
        "Accuracy": accuracy_score(y_val, predictions),
        "Precision": precision_score(y_val, predictions, zero_division=0),
        "Recall": recall_score(y_val, predictions, zero_division=0),
        "F1 Score": f1_score(y_val, predictions, zero_division=0),
        "ROC AUC": roc_auc_score(y_val, probabilities),
        "Average Precision": average_precision_score(y_val, probabilities),
    })


results_df = pd.DataFrame(results).sort_values(
    by="Accuracy",
    ascending=False,
)

print("\n===== FEATURE COMPARISON =====")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    "reports/feature_comparison.csv",
    index=False,
)

print("\nSaved to reports/feature_comparison.csv")

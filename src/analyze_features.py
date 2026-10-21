
import pandas as pd

from preprocessing import x_train, y_train

# Combine training features and target
df = x_train.copy()
df["Churn"] = y_train

# Numeric feature relationships with churn
numeric_cols = [
    col for col in [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "AvgMonthlySpend",
        "SeniorCitizen",
    ]
    if col in df.columns
]

print("\n===== NUMERIC FEATURE CORRELATIONS =====")
correlations = (
    df[numeric_cols]
    .corrwith(df["Churn"])
    .sort_values()
)

print(correlations.round(4).to_string())

# Churn rate across categories
categorical_cols = [
    col for col in [
        "Contract",
        "InternetService",
        "PaymentMethod",
        "OnlineSecurity",
        "TechSupport",
        "PaperlessBilling",
        "Partner",
        "Dependents",
    ]
    if col in df.columns
]

print("\n===== CHURN RATE BY CATEGORY =====")

for col in categorical_cols:
    print(f"\n--- {col} ---")

    summary = (
        df.groupby(col, observed=True)["Churn"]
        .agg(Customers="count", Churn_Rate="mean")
        .sort_values("Churn_Rate", ascending=False)
    )

    summary["Churn_Rate"] *= 100
    print(summary.round(2).to_string())

print("\nAnalysis complete.")

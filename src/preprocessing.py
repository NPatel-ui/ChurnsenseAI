import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
data_path=r"data\raw\Telco-Customer-Churn.csv"
df = pd.read_csv(data_path)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors='coerce')

# Feature engineering
df["AvgMonthlySpend"] = (
    df["TotalCharges"] / df["tenure"].replace(0, float("nan"))
)

df["AvgMonthlySpend"] = df["AvgMonthlySpend"].fillna(
    df["MonthlyCharges"]
)

# Count the services the customer subscribes to
service_columns = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

df["ServicesCount"] = sum(
    (df[col] == "Yes").astype(int)
    for col in service_columns
)

# Group tenure into lifecycle stages
df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, float("inf")],
    labels=["New", "Developing", "Established", "LongTerm"],
)

y=df["Churn"].map({"Yes": 1, "No": 0})
x=df.drop(columns=["Churn", "customerID"])
if y.isna().any():
    raise ValueError("Unexpected values found in churn target")
numeric_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "AvgMonthlySpend"
]
categorial_features=[
    col for col in x.columns
    if col not in numeric_features
]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.30, random_state=42,
)
x_val, x_test, y_val, y_test = train_test_split(
    x_test, y_test, test_size=0.50, random_state=42,
)
numeric_pipeline=Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
]
)
categorial_pipeline=Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
]
)
preprocessor=ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorial_pipeline, categorial_features),
])
print(y_train.value_counts(normalize=True).round(3))
print(y_val.value_counts(normalize=True).round(3))
print(y_test.value_counts(normalize=True).round(3))

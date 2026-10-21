import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
data_path=r"C:\Users\LENOVO\OneDrive\Desktop\Customer Churn AI\data\raw\Telco-Customer-Churn.csv"
figure_path=Path("reports/figures")
figure_path.mkdir(parents=True, exist_ok=True)
df=pd.read_csv(data_path)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors='coerce')
print(df.shape)
print(df.isnull().sum()[df.isnull().sum() > 0])
print(df.duplicated().sum())
print(df["customerID"].duplicated().sum())
print(pd.crosstab(df["Contract"], df["Churn"], normalize="index").round(3))
sns.set_theme(style="whitegrid")
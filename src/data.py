import pandas as pd
file_path=r"C:\Users\LENOVO\OneDrive\Desktop\Customer Churn AI\data\raw\Telco-Customer-Churn.csv"
df=pd.read_csv(file_path)
print(df.head())
print(df.shape)
print(df.columns.tolist())
df.info()
print(df.isnull().sum())
print(df.duplicated().sum())
print(df["Churn"].value_counts())
print(df["Churn"].value_counts(normalize=True).mul(100).round(2))
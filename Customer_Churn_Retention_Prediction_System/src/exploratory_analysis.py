import pandas as pd
df=pd.read_csv("data/processed/cleaned_telco_churn.csv")
print(df.describe(include="all").T)
print("\nChurn rate:",(df.Churn=="Yes").mean())

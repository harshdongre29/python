from pathlib import Path
import pandas as pd

RAW=Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
OUT=Path("data/processed/cleaned_telco_churn.csv")

def clean_data():
    df=pd.read_csv(RAW)
    df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce")
    df=df.dropna(subset=["TotalCharges"]).drop_duplicates().reset_index(drop=True)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(OUT,index=False)
    return df

if __name__=="__main__":
    print(clean_data().shape)

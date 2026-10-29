import argparse, joblib, pandas as pd
parser=argparse.ArgumentParser(); parser.add_argument("--customer_csv",default="data/processed/cleaned_telco_churn.csv"); args=parser.parse_args()
df=pd.read_csv(args.customer_csv); model=joblib.load("models/churn_model.joblib")
X=df.drop(columns=["Churn","customerID"],errors="ignore"); p=model.predict_proba(X)[:,1]
out=pd.DataFrame({"customerID":df["customerID"],"churn_probability":p})
out["risk_tier"]=pd.cut(out.churn_probability,[-.01,.30,.60,1.01],labels=["Low","Medium","High"])
out.to_csv("results/predictions.csv",index=False); print(out.head())

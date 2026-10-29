import pandas as pd
r=pd.read_csv("results/customer_risk_scores.csv")
targets=r[r.churn_probability>=.50].copy()
targets["recommended_action"]=targets.churn_probability.map(lambda x:"Immediate retention offer + personal outreach" if x>=.70 else "Targeted retention offer + service review")
targets.to_csv("results/retention_strategy_targets.csv",index=False)
print(f"Retention targets: {len(targets)}")

import pandas as pd
print(pd.read_csv("results/model_comparison.csv").to_string(index=False))
print("\nConfusion matrix:\n",pd.read_csv("results/confusion_matrix.csv",index_col=0))

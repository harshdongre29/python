from pathlib import Path
import pandas as pd
def test_dataset_exists():
    assert Path("data/processed/cleaned_telco_churn.csv").exists()
def test_target_exists():
    assert "Churn" in pd.read_csv("data/processed/cleaned_telco_churn.csv",nrows=2).columns
def test_model_exists():
    assert Path("models/churn_model.joblib").exists()

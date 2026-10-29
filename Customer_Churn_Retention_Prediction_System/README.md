# Customer Churn & Retention Prediction System

## Overview
A real-world machine learning project that predicts customer churn and converts churn probabilities into actionable retention priorities. The project uses the Telco Customer Churn dataset and provides a reproducible preprocessing, modeling, evaluation, scoring, and retention-analysis workflow.

## Problem
Telecom/customer-service businesses lose recurring revenue when customers leave. The goal is to identify customers with elevated churn risk early enough for targeted retention actions.

## Dataset
Source: Kaggle Telco Customer Churn dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`).
The supplied dataset contains 7,043 records and 21 columns. After converting `TotalCharges` to numeric, removing rows without usable total-charge values, and removing exact duplicate rows, the modeling dataset contains 7,032 customers. The target is `Churn`.

## Approach
1. Validate and clean raw customer data.
2. Convert `TotalCharges` to numeric and handle missing values.
3. Explore churn by contract, charges, tenure, services and demographics.
4. Encode categorical variables and scale numeric variables in a leakage-safe pipeline.
5. Train Logistic Regression and Random Forest baselines.
6. Compare Accuracy, Precision, Recall, F1, ROC-AUC and PR-AUC.
7. Use the selected Random Forest pipeline to generate customer-level churn probabilities.
8. Convert probabilities into Low, Medium and High risk tiers.
9. Estimate monthly revenue exposure and produce retention targets.

## Model Results
The held-out test set contains 20% of the cleaned dataset, stratified by churn.
Random Forest test metrics: Accuracy **0.768**, Precision **0.548**, Recall **0.717**, F1 **0.621**, ROC-AUC **0.831**, PR-AUC **0.633**.

These are offline evaluation results on this dataset, not guarantees about future customer behavior.

## Business Outputs
- `results/customer_risk_scores.csv`: churn probability and risk tier for each customer.
- `results/retention_targets.csv`: customers at or above the priority threshold.
- `results/revenue_at_risk.csv`: aggregate monthly revenue exposure estimate.
- `results/feature_importance.csv`: top model features.

## Retention Strategy
High-risk customers can be prioritized for personal outreach, service review, contract incentives or targeted offers. Medium-risk customers can receive preventive engagement. Low-risk customers can remain in routine retention programs. Actions should be validated with business costs and customer experience constraints.

## Project Structure
```text
data/          raw and processed data
notebooks/     analysis notebook
src/           preprocessing, training, prediction and strategy scripts
models/        trained model
results/       metrics and business outputs
visualizations/ charts
app/            optional dashboard entry point
tests/          basic project tests
```

## Installation
Requires Python 3.10+.

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python src/data_preprocessing.py
python src/train_model.py
python src/evaluate_model.py
python src/retention_strategy.py
python src/predict.py
```

## Data and Privacy
The repository contains the project dataset supplied for this educational analysis. Do not add private customer information, credentials, API keys, or secrets.

## License
MIT License.

from pathlib import Path
import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

DATA=Path("data/processed/cleaned_telco_churn.csv")
df=pd.read_csv(DATA)
X=df.drop(columns=["Churn","customerID"]); y=(df["Churn"]=="Yes").astype(int)
cat=X.select_dtypes(include="object").columns.tolist(); num=X.select_dtypes(exclude="object").columns.tolist()
pre=ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]),num),
("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]),cat)])
model=Pipeline([("pre",pre),("model",RandomForestClassifier(n_estimators=350,max_depth=12,min_samples_leaf=3,class_weight="balanced",random_state=42,n_jobs=-1))])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
model.fit(Xtr,ytr)
Path("models").mkdir(exist_ok=True); joblib.dump(model,"models/churn_model.joblib")
print("Model saved.")

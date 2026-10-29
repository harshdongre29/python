import streamlit as st, pandas as pd, joblib
st.set_page_config(page_title="Customer Churn & Retention",layout="wide")
st.title("Customer Churn & Retention Prediction System")
st.write("Upload a customer CSV using the same feature columns as the training dataset.")
file=st.file_uploader("Customer CSV",type="csv")
if file:
    df=pd.read_csv(file); model=joblib.load("models/churn_model.joblib")
    X=df.drop(columns=["Churn","customerID"],errors="ignore")
    p=model.predict_proba(X)[:,1]
    out=pd.DataFrame({"customerID":df.get("customerID",range(len(df))),"churn_probability":p})
    out["risk_tier"]=pd.cut(out.churn_probability,[-.01,.30,.60,1.01],labels=["Low","Medium","High"])
    st.metric("Customers",len(out)); st.metric("High Risk",(out.risk_tier=="High").sum())
    st.dataframe(out.sort_values("churn_probability",ascending=False),use_container_width=True)

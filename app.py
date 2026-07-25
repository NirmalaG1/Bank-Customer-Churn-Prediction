from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app=FastAPI(title="Bank Customer Churn Prediction API")

model=joblib.load("xgb_model.pkl")
scaler=joblib.load("scaler.pkl")
model_columns=joblib.load("model_columns.pkl")

class CustomerData(BaseModel):
    Credit_Score:int
    Age:int
    Tenure:int
    Balance:float
    NumOfProducts:int
    HasCreditCard:int # 1 or 0
    IsActiveMember:int # 1 or 0
    EstimatedSalary:float
    Country:str #options:'France','Germany','Spain'
    Gender:str #options:'Male','Female'

@app.get("/")
def home():
    return {"message": "Bank Customer Churn Prediction API is running"}

@app.post("/predict")
def predict_churn(data:CustomerData):
    

 if data.Country == "Germany":
        
        Country_Germany = 1
        Country_Spain = 0

 elif data.Country == "Spain":
        Country_Germany = 0
        Country_Spain = 1
 else:
       
        Country_Germany = 0
        Country_Spain = 0

 Gender_male = 1 if data.Gender == "Male" else 0

 input_data = {
        "Credit_Score": data.Credit_Score,
        "Age": data.Age,
        "Tenure": data.Tenure,
        "Balance": data.Balance,
        "NumOfProducts": data.NumOfProducts,
        "HasCreditCard": data.HasCreditCard,
        "IsActiveMember": data.IsActiveMember,
        "EstimatedSalary": data.EstimatedSalary,
        "Country_Germany": Country_Germany,
        "Country_Spain": Country_Spain,
        "Gender_Male": Gender_male
    }

 num_df=pd.DataFrame([{
             "Credit_Score": data.Credit_Score,
              "Age": data.Age,
              "Tenure": data.Tenure,
              "Balance": data.Balance,
              "NumOfProducts": data.NumOfProducts,
              "HasCreditCard": data.HasCreditCard,
              "IsActiveMember": data.IsActiveMember,
              "EstimatedSalary": data.EstimatedSalary
 }])
 scaled_df=pd.DataFrame(scaler.transform(num_df),columns=num_df.columns)

 cat_df=pd.DataFrame([{
         "Country_Germany": Country_Germany,
         "Country_Spain": Country_Spain,
         "Gender_Male": Gender_male
 }])
     

 df=pd.concat([scaled_df,cat_df],axis=1)
 df=df[model_columns]
 
 prediction = model.predict(df)[0]
 return {
        "prediction": int(prediction),
        "status": "Churned " if prediction == 1 else "Not churned",
    }

    

 
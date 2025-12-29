from fastapi import FastAPI
import pandas as pd

app = FastAPI()

df = pd.read_csv("ml_output.csv")

@app.get("/")
def root():
    return {"message": "Battery Monitoring API is running"}

@app.get("/data")
def get_data():
    return df.to_dict(orient="records")

@app.get("/anomalies")
def get_anomalies():
    anomalies = df[df["final_anomaly"] == 1]
    return anomalies.to_dict(orient="records")

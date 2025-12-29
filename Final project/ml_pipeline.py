import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ==============================
# 1. LOAD DATA
# ==============================
CSV_PATH = "Final project/HRD_Data.csv"   # <-- ensure correct file name

df = pd.read_csv(CSV_PATH)
print("\n📄 Columns:", df.columns.tolist())

# ==============================
# 2. HANDLE TIMESTAMP PROPERLY
# ==============================
df.rename(columns={"TaktTime": "timestamp"}, inplace=True)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

print("⏱️ Timestamp column fixed")

# ==============================
# 3. SELECT CORRECT FEATURES
# ==============================
df.rename(columns={
    "Voltage": "voltage",
    "Mosfet Body Temp": "temperature"
}, inplace=True)

features = ["voltage", "temperature"]

for f in features:
    df[f] = pd.to_numeric(df[f], errors="coerce")

df = df.dropna(subset=features)

print("📊 Data shape after cleaning:", df.shape)

# ==============================
# 4. RULE-BASED THRESHOLDS
# ==============================
THRESHOLDS = {
    "voltage": {"min": 48, "max": 52},
    "temperature": {"max": 25}
}

df["rule_anomaly"] = 0

df.loc[
    (df["voltage"] < THRESHOLDS["voltage"]["min"]) |
    (df["voltage"] > THRESHOLDS["voltage"]["max"]),
    "rule_anomaly"
] = 1

df.loc[
    df["temperature"] > THRESHOLDS["temperature"]["max"],
    "rule_anomaly"
] = 1

# ==============================
# 5. ML – ISOLATION FOREST
# ==============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)

df["ml_raw"] = model.fit_predict(X_scaled)
df["ml_anomaly"] = (df["ml_raw"] == -1).astype(int)

# ==============================
# 6. FINAL ANOMALY
# ==============================
df["final_anomaly"] = (
    (df["rule_anomaly"] == 1) |
    (df["ml_anomaly"] == 1)
).astype(int)

# ==============================
# 7. SAVE OUTPUT
# ==============================
output = df[
    ["timestamp", "voltage", "temperature",
     "rule_anomaly", "ml_anomaly", "final_anomaly"]
]

output.to_csv("Final project/ml_output.csv", index=False)

print("\n✅ ML PIPELINE COMPLETED")
print("Rule anomalies :", output["rule_anomaly"].sum())
print("ML anomalies   :", output["ml_anomaly"].sum())
print("Final anomalies:", output["final_anomaly"].sum())
print("📁 Saved: ml_output.csv")


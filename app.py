import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------
# PV Detective - Simple Solar PV Analysis
# ---------------------------------------

FILE_PATH = "data/Dataset-SolarTechLab.csv"

# Load dataset
df = pd.read_csv(FILE_PATH, sep=";")

# Convert columns
df["Time"] = pd.to_datetime(df["Time"], errors="coerce")

numeric_columns = ["PV_Power", "T_air", "G_h", "G_tilt", "W_s", "W_d"]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Replace invalid sensor values
df.loc[df["T_air"] <= -100, "T_air"] = np.nan

# Remove invalid timestamps
df = df.dropna(subset=["Time"]).sort_values("Time")

# Remove negative power values
df.loc[df["PV_Power"] < 0, "PV_Power"] = np.nan

# Estimate energy from one-minute readings
df["Energy_kWh"] = df["PV_Power"] / 1000 / 60

# Create daily summary
df["Date"] = df["Time"].dt.date

daily_summary = df.groupby("Date").agg(
    Total_Energy_kWh=("Energy_kWh", "sum"),
    Average_Power_W=("PV_Power", "mean"),
    Maximum_Power_W=("PV_Power", "max"),
    Average_Temperature=("T_air", "mean"),
    Average_Irradiance=("G_tilt", "mean")
).reset_index()

# Simple anomaly rule:
# High irradiance but unusually low power
power_limit = df["PV_Power"].quantile(0.10)
df["Anomaly"] = (
    (df["G_tilt"] > 200) &
    (df["PV_Power"] < power_limit)
)

# ---------------------------------------
# Dashboard
# ---------------------------------------

print("\n" + "=" * 55)
print("              PV DETECTIVE")
print("      Solar PV Performance Analysis")
print("=" * 55)

print("\nDATASET SUMMARY")
print("-" * 55)
print("Total records:", len(df))
print("Start date:", df["Time"].min())
print("End date:", df["Time"].max())
print("Total missing values:", df.isna().sum().sum())

print("\nPERFORMANCE SUMMARY")
print("-" * 55)
print(f"Total estimated energy: {df['Energy_kWh'].sum():.2f} kWh")
print(f"Average power: {df['PV_Power'].mean():.2f} W")
print(f"Maximum power: {df['PV_Power'].max():.2f} W")
print(f"Average temperature: {df['T_air'].mean():.2f} °C")
print(f"Average tilted irradiance: {df['G_tilt'].mean():.2f} W/m²")
print("Unusual observations:", int(df["Anomaly"].sum()))

# ---------------------------------------
# Graph 1: Power over time
# ---------------------------------------

plt.figure(figsize=(12, 5))
plt.plot(df["Time"], df["PV_Power"], linewidth=0.7)
plt.title("PV Power Over Time")
plt.xlabel("Time")
plt.ylabel("PV Power (W)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# ---------------------------------------
# Graph 2: Irradiance vs Power
# ---------------------------------------

plt.figure(figsize=(8, 5))
plt.scatter(df["G_tilt"], df["PV_Power"], s=4)
plt.title("Tilted Irradiance vs PV Power")
plt.xlabel("Tilted Irradiance (W/m²)")
plt.ylabel("PV Power (W)")
plt.tight_layout()
plt.show()

# ---------------------------------------
# Graph 3: Temperature vs Power
# ---------------------------------------

plt.figure(figsize=(8, 5))
plt.scatter(df["T_air"], df["PV_Power"], s=4)
plt.title("Air Temperature vs PV Power")
plt.xlabel("Air Temperature (°C)")
plt.ylabel("PV Power (W)")
plt.tight_layout()
plt.show()

# ---------------------------------------
# Graph 4: Daily energy
# ---------------------------------------

plt.figure(figsize=(12, 5))
plt.plot(
    daily_summary["Date"],
    daily_summary["Total_Energy_kWh"]
)
plt.title("Daily Energy Generation")
plt.xlabel("Date")
plt.ylabel("Energy (kWh)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# Save results
daily_summary.to_csv("daily_pv_summary.csv", index=False)
df.to_csv("analyzed_pv_data.csv", index=False)

print("\nAnalysis completed successfully.")
print("Files created: daily_pv_summary.csv and analyzed_pv_data.csv")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# NASA Global Temperature Data
# --------------------------------------------------

temp_url = (
    "https://data.giss.nasa.gov/gistemp/tabledata_v4/"
    "GLB.Ts+dSST.csv"
)

temp_df = pd.read_csv(temp_url, skiprows=1)

temp_df = temp_df[["Year", "J-D"]]

temp_df["Year"] = pd.to_numeric(temp_df["Year"], errors="coerce")
temp_df["J-D"] = pd.to_numeric(temp_df["J-D"], errors="coerce")

temp_df = temp_df.dropna()

# Convert anomaly to approximate absolute temperature

baseline_temp_F = 57.2   # 14°C

temp_df["Temp_F"] = baseline_temp_F + temp_df["J-D"] * 9.0 / 5.0

# --------------------------------------------------
# Global CO2 Emissions Data
# --------------------------------------------------

co2_url = (
    "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
)

co2_df = pd.read_csv(co2_url)

# World totals only
co2_df = co2_df[co2_df["country"] == "World"]

co2_df = co2_df[["year", "co2"]]

co2_df.rename(columns={"year": "Year"}, inplace=True)

# --------------------------------------------------
# Merge datasets
# --------------------------------------------------

df = pd.merge(
    temp_df[["Year", "Temp_F"]],
    co2_df,
    on="Year",
    how="inner"
)

df = df[df["Year"] >= 1950]

# --------------------------------------------------
# Linear fit
# --------------------------------------------------

slope, intercept = np.polyfit(
    df["co2"],
    df["Temp_F"],
    1
)

xfit = np.linspace(
    df["co2"].min(),
    df["co2"].max(),
    200
)

yfit = slope * xfit + intercept

# --------------------------------------------------
# Scatter plot
# --------------------------------------------------

plt.figure(figsize=(10, 7))

plt.scatter(
    df["co2"],
    df["Temp_F"],
    alpha=0.8,
    color="blue"
)

plt.plot(
    xfit,
    yfit,
    color="red",
    linewidth=3,
    label=f"Fit: T = {slope:.4f}·CO₂ + {intercept:.2f}"
)

plt.xlabel("Global CO₂ Emissions (Gt CO₂/year)")
plt.ylabel("Global Mean Temperature (°F)")
plt.title("Global Temperature vs Global CO₂ Emissions\n(1950-Present)")
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

# --------------------------------------------------
# Correlation
# --------------------------------------------------

corr = df["co2"].corr(df["Temp_F"])

print()
print(f"Correlation coefficient = {corr:.3f}")
print(f"Slope = {slope:.4f} °F per Gt CO₂")
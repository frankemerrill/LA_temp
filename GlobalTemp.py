import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# NASA GISTEMP annual global temperature anomalies
# --------------------------------------------------

url = (
    "https://data.giss.nasa.gov/gistemp/tabledata_v4/"
    "GLB.Ts+dSST.csv"
)

# Read NASA table
df = pd.read_csv(url, skiprows=1)

# Keep year and annual anomaly
df = df[["Year", "J-D"]]

# Convert to numeric
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["J-D"] = pd.to_numeric(df["J-D"], errors="coerce")

# Remove missing values
df = df.dropna()

# Keep 1950 onward
df = df[df["Year"] >=1900]

# --------------------------------------------------
# Convert anomaly to approximate absolute temperature
# --------------------------------------------------
#
# NASA anomalies are relative to the 1951-1980 mean.
# Approximate global mean temperature during that
# period was about 14.0 °C.
#
baseline_temp = 14.0

df["GlobalTemp"] = baseline_temp + df["J-D"]

## Convert to Fahrenheit
df["GlobalTemp_F"] = df["GlobalTemp"] * 9/5 + 32

# --------------------------------------------------
# Plot
# --------------------------------------------------

plt.figure(figsize=(12,6))

plt.plot(
    df["Year"],
    df["GlobalTemp_F"],
    color="blue",
    linewidth=2
)

plt.xlabel("Year")
plt.ylabel("Global Mean Temperature (°F)")
plt.title("Global Mean Surface Temperature (NASA GISTEMP)")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

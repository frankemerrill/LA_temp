import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Read temperature data
# --------------------------------------------------

filename = "LA_Temp.csv"

df = pd.read_csv(
    filename,
    header=None,
    names=[
        "Station",
        "Location",
        "Date",
        "Tmax",
        "Tmin",
        "Temp"
    ]
)

# Convert columns to appropriate types
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Tmax"] = pd.to_numeric(df["Tmax"], errors="coerce")
df["Tmin"] = pd.to_numeric(df["Tmin"], errors="coerce")
df["Temp"] = pd.to_numeric(df["Temp"], errors="coerce")

# Remove rows with missing temperatures or invalid dates
df = df.dropna(subset=["Date", "Tmax", "Tmin", "Temp"])

# --------------------------------------------------
# Extract July observations
# --------------------------------------------------

july_df = df[df["Date"].dt.month == 7].copy()

# Extract year
july_df["Year"] = july_df["Date"].dt.year
july_df["Month"] = july_df["Date"].dt.month
july_df["Day"] = july_df["Date"].dt.day

# --------------------------------------------------
# Calculate annual July statistics
# --------------------------------------------------

july_stats = (
    july_df.groupby("Year")["Tmax"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

july_stats.rename(
    columns={
        "mean": "JulyMeanTemp",
        "std": "JulyStdTemp",
        "count": "NumObs"
    },
    inplace=True
)

print("\nJuly Statistics by Year\n")
#print(july_stats)

# If Year is the index
years = july_stats.index

years = july_stats["Year"]

plt.figure(figsize=(10,6))

plt.errorbar(
    years,
    july_stats["JulyMeanTemp"],
    yerr=july_stats["JulyStdTemp"],
    fmt='o',
    capsize=5,
    linewidth=2,
    markersize=6,
    ecolor='red'
)

plt.xlabel("Year")
plt.ylabel("Temperature")
plt.title("Mean July Temperature by Year")
plt.grid(True)

# --------------------------------------------------
# Extract January observations
# --------------------------------------------------

jan_df = df[df["Date"].dt.month == 1].copy()

# Extract year
jan_df["Year"] = jan_df["Date"].dt.year

# --------------------------------------------------
# Calculate annual January low-temperature statistics
# --------------------------------------------------

jan_stats = (
    jan_df.groupby("Year")["Tmin"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

jan_stats.rename(
    columns={
        "mean": "JanMeanLowTemp",
        "std": "JanStdLowTemp",
        "count": "NumObs"
    },
    inplace=True
)

print("\nJanuary Low Temperature Statistics by Year\n")
print(jan_stats)

# --------------------------------------------------
# Plot Average January Low Temperature
# with standard deviation error bars
# --------------------------------------------------

years = jan_stats["Year"]

plt.figure(figsize=(10,6))

plt.errorbar(
    years,
    jan_stats["JanMeanLowTemp"],
    yerr=jan_stats["JanStdLowTemp"],
    fmt='o',
    capsize=5,
    linewidth=2,
    markersize=6,
    ecolor='red'
)

plt.xlabel("Year")
plt.ylabel("Average January Low Temperature")
plt.title("Average January Low Temperature by Year")
plt.grid(True)

plt.tight_layout()
plt.show()

# Print July 2003 high temperatures

#july_2003_highs = july_df[july_df["Year"] == 2003][["Date", "Tmax"]]

#print("\nJuly 2003 High Temperatures\n")
#print(july_2003_highs.to_string(index=False))
# -------------------
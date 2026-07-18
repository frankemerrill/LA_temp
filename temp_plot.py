import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


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
        "TempObs"
    ]
)

# Convert columns to appropriate types
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Tmax"] = pd.to_numeric(df["Tmax"], errors="coerce")
df["Tmin"] = pd.to_numeric(df["Tmin"], errors="coerce")
df["TempObs"] = pd.to_numeric(df["TempObs"], errors="coerce")

# Remove rows with missing temperatures or invalid dates
df = df.dropna(subset=["Date", "Tmax", "Tmin", "TempObs"])

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
        "mean": "JulyMeanMaxTemp",
        "std": "JulyStdMaxTemp",
        "count": "NumObs"
    },
    inplace=True
)

years = july_stats["Year"]

plt.figure(figsize=(10,6))

plt.errorbar(
    years,
    july_stats["JulyMeanMaxTemp"],
    yerr=july_stats["JulyStdMaxTemp"],
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
print(july_stats)

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


# --------------------------------------------------
# July Mean Maximum Temperature
# with Rolling Average
# --------------------------------------------------

# Calculate 3-year rolling average of July mean maximum temperature
july_stats["RollingAvg"] = (
    july_stats["JulyMeanMaxTemp"]
    .rolling(window=5, center=True)
    .mean()
)

# Plot July mean maximum temperature and rolling average
plt.figure(figsize=(10,6))

plt.plot(
    july_stats["Year"],
    july_stats["JulyMeanMaxTemp"],
    marker="o",
    linestyle="-",
    color="lightgray",
    linewidth=1,
    markersize=4,
    label="July Mean Maximum Temperature"
)

plt.plot(
    july_stats["Year"],
    july_stats["RollingAvg"],
    color="blue",
    linewidth=3,
    label="Rolling Average"
)

plt.xlabel("Year")
plt.ylabel("Temperature")
plt.title("July Mean Maximum Temperature\n with Rolling Average")
plt.grid(True)
plt.legend()

# --------------------------------------------------
# January Mean lowTemperature
# with Rolling Average
# --------------------------------------------------

# Calculate 3-year rolling average of January mean low temperature
jan_stats["RollingAvg"] = (
    jan_stats["JanMeanLowTemp"]
    .rolling(window=5, center=True)
    .mean()
)

# Plot July mean maximum temperature and rolling average
plt.figure(figsize=(10,6))

plt.plot(
    jan_stats["Year"],
    jan_stats["JanMeanLowTemp"],
    marker="o",
    linestyle="-",
    color="lightgray",
    linewidth=1,
    markersize=4,
    label="January Mean Low Temperature"
)

plt.plot(
    jan_stats["Year"],
    jan_stats["RollingAvg"],
    color="blue",
    linewidth=3,
    label="Rolling Average"
)

plt.xlabel("Year")
plt.ylabel("Temperature")
plt.title("January Mean Low Temperature\n with Rolling Average")
plt.grid(True)
plt.legend()

# Create histogram of July mean max temperatures
plt.figure(figsize=(8, 6))

plt.hist(
    july_stats["JulyMeanMaxTemp"],
    bins=20,           # adjust as desired
    color="skyblue",
    edgecolor="black",
    alpha=0.7
)

# Add vertical line at 84°F
plt.axvline(
    x=84,
    color="red",
    linestyle="--",
    linewidth=2,
    label="84°F"
)

plt.xlabel("July Mean Max Temperature (°F)")
plt.ylabel("Frequency")
plt.title("Distribution of July Mean Max Temperatures")
plt.legend()
plt.grid(alpha=0.3)

# --------------------------------------------------
# Extract temperature values and remove NaNs if necessary
# --------------------------------------------------

temps = july_stats["JulyMeanMaxTemp"].dropna()

# Fit Gaussian (estimate mean and standard deviation)
mu, sigma = norm.fit(temps)

# Probability that temperature is >= 84 F
threshold = 83.5
p_ge_84 = norm.sf(threshold, loc=mu, scale=sigma)  # survival function = 1 - CDF

print(f"Gaussian fit:")
print(f"  Mean = {mu:.2f} F")
print(f"  Std Dev = {sigma:.2f} F")
print(f"  P(T >= 83.5 F) = {p_ge_84:.4f}")
print(f"  P(T >= 83.5 F) = {100*p_ge_84:.2f}%")

# Create figure
fig, ax = plt.subplots(figsize=(8, 6))

# Normalized histogram (area = 1)
counts, bins, patches = ax.hist(
    temps,
    bins=20,
    density=True,
    alpha=0.6,
    color="skyblue",
    edgecolor="black",
    label="Observed temperatures"
)

# Gaussian PDF
x = np.linspace(min(temps), max(temps), 500)
pdf = norm.pdf(x, mu, sigma)

ax.plot(
    x,
    pdf,
    'r-',
    linewidth=2.5,
    label=f'Gaussian Fit\nμ={mu:.2f}, σ={sigma:.2f}'
)

# Vertical line at 84 F
ax.axvline(
    threshold,
    color='black',
    linestyle='--',
    linewidth=2,
    label='84°F'
)

# Optionally shade the tail corresponding to P(T >= 84)
x_tail = np.linspace(threshold, max(x), 300)
ax.fill_between(
    x_tail,
    norm.pdf(x_tail, mu, sigma),
    alpha=0.3,
    color='red',
    label=f'P(T ≥ 84) = {100*p_ge_84:.2f}%'
)

ax.set_xlabel("July Mean Max Temperature (°F)")
ax.set_ylabel("Probability Density")
ax.set_title("Distribution of July Mean Max Temperatures")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
# -------------------
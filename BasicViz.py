import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import os
from lifelines import KaplanMeierFitter




#############################
###### VIZUALIZATION 1 ######
#############################

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
        
booking = pd.read_csv('cleaned_dataset.csv')

numeric_booking = booking.select_dtypes(include=['number'])  # Keep only numeric columns

plt.figure(figsize=(35,12))
sns.heatmap(numeric_booking.corr(), annot=True, linecolor='black', linewidths=2, cmap='rocket', fmt=".2f")

plt.subplots_adjust(left=0.2)
plt.subplots_adjust(bottom=0.2)
plt.xticks(rotation=45, ha='right')

plt.show()


#############################
###### VIZUALIZATION 2 ######
#############################

# Set style
sns.set_style("whitegrid")

plt.figure(figsize=(8, 6))  # Adjusted figure size


# Draw countplot
ax = sns.countplot(
    x='is_canceled', 
    data=booking, 
    palette="rocket",  # Color theme
    edgecolor="black"   # Add border to bars
)

# Add labels on top of bars without decimals
for p in ax.patches:
    ax.annotate(
        f'{int(p.get_height()):,}',  # Convert to integer and format with commas
        (p.get_x() + p.get_width() / 2., p.get_height()), 
        ha='center', va='bottom',
        fontsize=12, fontweight="bold", color="black"
    )

# Format y-axis in thousands (K)
ax.get_yaxis().set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))

# Customize axes and title
plt.xlabel("", fontsize=14, fontweight="bold")
plt.ylabel("Number of Bookings", fontsize=14, fontweight="bold")
plt.title("Hotel Booking Cancellations", fontsize=16, fontweight="bold", pad=20)  # Increase title spacing
plt.xticks([0, 1], ["Not Canceled", "Canceled"], fontsize=12)  # Rename x labels

# Remove unnecessary borders
sns.despine()

# Show the plot
plt.show()

#############################
###### VIZUALIZATION 3 ######
#############################

# Set Seaborn style
sns.set_style("whitegrid")

# Count cancellations and non-cancellations per hotel type
hotel_cancellation_counts = booking.groupby(["hotel", "is_canceled"]).size().unstack()

# Plot bar chart using Seaborn
ax = hotel_cancellation_counts.plot(
    kind='bar',
    figsize=(8, 6),
    color=sns.color_palette("rocket", 2),  # Use rocket color palette
    edgecolor="black"
)

# Add labels on top of bars (without decimals)
for p in ax.patches:
    ax.annotate(
        f'{int(p.get_height()):,}',  # Convert to integer with commas
        (p.get_x() + p.get_width() / 2., p.get_height()), 
        ha='center', va='bottom',
        fontsize=12, fontweight="bold", color="black"
    )

# Format y-axis in thousands (K)
ax.get_yaxis().set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))

# Customize labels and title
plt.xlabel("Hotel Type", fontsize=14, fontweight="bold")
plt.ylabel("Number of Bookings", fontsize=14, fontweight="bold")
plt.title("Hotel Booking Cancellations by Hotel Type", fontsize=16, fontweight="bold", pad=20)  # Extra space in title
plt.xticks(rotation=0, fontsize=12)  # Keep hotel type labels horizontal
plt.legend(["Not Canceled", "Canceled"], fontsize=12, title="Booking Status")

# Remove unnecessary borders
sns.despine()

# Show plot
plt.show()


#############################
###### VIZUALIZATION 4 ######
#############################

# Set Seaborn style
sns.set_style("whitegrid")

# Ensure reservation_status_date is a datetime format
booking["reservation_status_date"] = pd.to_datetime(booking["reservation_status_date"])

# Filter data for each hotel type and clean anomalies
resort_hotel = booking[booking["hotel"] == "Resort Hotel"].groupby("reservation_status_date")[["adr"]].mean()
city_hotel = booking[booking["hotel"] == "City Hotel"].groupby("reservation_status_date")[["adr"]].mean()

# Drop possible anomalies (e.g., ADR = 0, NaN)
resort_hotel = resort_hotel[resort_hotel["adr"] > 0].dropna()
city_hotel = city_hotel[city_hotel["adr"] > 0].dropna()

# Apply 7-day moving average to smooth the data
resort_hotel["adr_smooth"] = resort_hotel["adr"].rolling(window=7, min_periods=1).mean()
city_hotel["adr_smooth"] = city_hotel["adr"].rolling(window=7, min_periods=1).mean()

# Create figure
plt.figure(figsize=(20, 8))

# Plot ADR trends with smoothed lines
plt.plot(resort_hotel.index, resort_hotel["adr_smooth"], label="Resort Hotel", 
         color=sns.color_palette("rocket")[0], linewidth=3)
plt.plot(city_hotel.index, city_hotel["adr_smooth"], label="City Hotel", 
         color=sns.color_palette("rocket")[2], linewidth=3)

# Formatting
plt.title("Average Daily Rate (7-Day Moving Average)", 
          fontsize=28, fontweight="bold", pad=20)
plt.ylabel("Average Daily Rate (€)", fontsize=16, fontweight="bold")

# Improve x-axis date formatting
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # Show every 6 months
plt.xticks(rotation=30, fontsize=14)
plt.yticks(fontsize=14) 

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))  # Format as "Mar 2025"

# Customize legend
plt.legend(fontsize=12, title="Hotel Type", title_fontsize=14)

# Add light grid for readability
plt.grid(alpha=0.3)

# Show plot
plt.show()


#############################
###### VIZUALIZATION 5 ######
#############################


# Copy dataset to avoid modifying the original DataFrame
booking_viz5 = booking.copy()

# Filter only canceled bookings
booking_viz5 = booking_viz5[booking_viz5["is_canceled"] == 1]

# Use lead_time as the time until cancellation
booking_viz5["days_until_cancellation"] = booking_viz5["lead_time"]

# Ensure no negative values (data errors)
booking_viz5 = booking_viz5[booking_viz5["days_until_cancellation"] >= 0]

# Define ADR bins dynamically
adr_bins = np.arange(0, booking_viz5["adr"].max() + 15, 20)  
booking_viz5["adr_bin"] = pd.cut(booking_viz5["adr"], bins=adr_bins, right=False)

# Drop NaN ADR bins
booking_viz5 = booking_viz5.dropna(subset=["adr_bin"])

# === Dynamic Minimum Count Calculation ===
percentage_threshold = 0.01  # 0.5% of total bookings
min_count = max(30, int(percentage_threshold * len(booking_viz5)))  # Ensure reasonable min

# Count number of bookings per ADR bin
bin_counts = booking_viz5["adr_bin"].value_counts()

# Compute dynamic threshold based on median bin size
adaptive_threshold = max(30, int(0.5 * bin_counts.median()))  # Take at least 50% of median bin size

# Take the **larger** of the two dynamic thresholds
final_min_count = max(min_count, adaptive_threshold)

# Keep only bins with enough data points
valid_bins = bin_counts[bin_counts >= final_min_count].index

# Filter dataset to only include valid bins
booking_viz5 = booking_viz5[booking_viz5["adr_bin"].isin(valid_bins)]

# Convert ADR bins to integer-friendly labels (e.g., "50-100" instead of "[50.0, 100.0]")
booking_viz5["adr_bin"] = booking_viz5["adr_bin"].apply(lambda x: f"{int(x.left)}-{int(x.right)}")

# Sort ADR bins numerically
sorted_bins = sorted(booking_viz5["adr_bin"].unique(), key=lambda x: int(x.split("-")[0]))

# Set up the figure
fig, ax1 = plt.subplots(figsize=(12, 6))
kmf = KaplanMeierFitter()

# Dictionary to store percentage remaining at each time step
time_points = np.arange(0, booking_viz5["days_until_cancellation"].max(), 5)  # Every 5 days
percent_remaining = np.zeros_like(time_points, dtype=float)

# Loop through ADR bins in sorted order
for adr_bin in sorted_bins:
    subset = booking_viz5[booking_viz5["adr_bin"] == adr_bin]
    if len(subset) >= final_min_count:  # Double check to avoid empty subsets
        kmf.fit(subset["days_until_cancellation"], event_observed=subset["is_canceled"], label=adr_bin)
        kmf.plot_survival_function(ax=ax1)

# Compute percentage of data remaining at each time point
total_count = len(booking_viz5)
for i, t in enumerate(time_points):
    remaining_count = len(booking_viz5[booking_viz5["days_until_cancellation"] >= t])
    percent_remaining[i] = (remaining_count / total_count) * 100

# Create secondary y-axis for percentage of data remaining
ax2 = ax1.twinx()
ax2.plot(time_points, percent_remaining / 100, color="black", linestyle="dashed", linewidth=2, label="Data Remaining (%)")  # Normalize to match survival scale
ax2.set_ylabel("Percentage of Data Remaining", fontsize=12, fontweight="bold", color="black")

# === 🔥 Align both y-axes so that 1.0 (survival) = 100% (data remaining) ===
ax1.set_ylim(0, 1)  # Survival probability from 0 to 1
ax2.set_ylim(0, 1)  # Convert percentage to 0-1 scale (100% = 1.0)

# Formatting
ax1.set_xlabel("Days Since Booking (Lead Time)", fontsize=14, fontweight="bold")
ax1.set_ylabel("Survival Probability (Not Canceled)", fontsize=14, fontweight="bold")
ax1.set_title("Survival Curve: Cancellation Probability by ADR", fontsize=16, fontweight="bold")

# === Fix legend order ===
handles, labels = ax1.get_legend_handles_labels()
sorted_legend = sorted(zip(labels, handles), key=lambda x: int(x[0].split("-")[0]))  # Sort numerically
labels, handles = zip(*sorted_legend)  # Unpack sorted tuples
ax1.legend(handles, labels, title="ADR (€)", fontsize=10)  # Reapply legend in sorted order

# Add legend for percentage line
ax2.legend(loc="center right")

# Make the gridlines align between both y-axes
ax1.grid(True, linestyle="--", alpha=0.5)  # Enable grid only for primary y-axis
ax2.grid(False)  # Disable separate grid for secondary y-axis

plt.xticks(rotation=45)

# Show the plot
plt.show()
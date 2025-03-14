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
        
        
booking = pd.read_csv('/Users/lasserathke/Desktop/Universität/Python/Hotel Dataset/Data/cleaned_dataset.csv')

numeric_booking = booking.select_dtypes(include=['number'])

plt.figure(figsize=(35,12))
sns.heatmap(numeric_booking.corr(), annot=True, linecolor='black', linewidths=2, cmap='rocket', fmt=".2f")

plt.subplots_adjust(left=0.2)
plt.subplots_adjust(bottom=0.2)
plt.xticks(rotation=45, ha='right')


plt.show()


#############################
###### VIZUALIZATION 2 ######
#############################

sns.set_style("whitegrid")

plt.figure(figsize=(8, 6))

ax = sns.countplot(
    x='is_canceled', 
    data=booking, 
    palette="rocket", 
    edgecolor="black"
)

for p in ax.patches:
    p.set_width(p.get_width() * 0.5)
    p.set_x(p.get_x() + p.get_width() / 2)

for p in ax.patches:
    ax.annotate(
        f'{int(p.get_height()):,}',
        (p.get_x() + p.get_width() / 2., p.get_height()), 
        ha='center', va='bottom',
        fontsize=12, fontweight="bold", color="black"
    )

# Format y-axis in thousands (K)
ax.get_yaxis().set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))

plt.xlabel("", fontsize=14, fontweight="bold")
plt.ylabel("Number of Bookings", fontsize=14, fontweight="bold")
plt.xticks([0, 1], ["Not Canceled", "Canceled"], fontsize=12)

sns.despine()

plt.grid(alpha=0)

plt.show()

#############################
###### VIZUALIZATION 3 ######
#############################

sns.set_style("whitegrid")

# reservation_status_date to datetime format
booking["reservation_status_date"] = pd.to_datetime(booking["reservation_status_date"])

# Filter + clean anomalies
resort_hotel = booking[booking["hotel"] == "Resort Hotel"].groupby("reservation_status_date")[["adr"]].mean()
city_hotel = booking[booking["hotel"] == "City Hotel"].groupby("reservation_status_date")[["adr"]].mean()

resort_hotel = resort_hotel[resort_hotel["adr"] > 0].dropna()
city_hotel = city_hotel[city_hotel["adr"] > 0].dropna()

# 7-day moving average to smooth data
resort_hotel["adr_smooth"] = resort_hotel["adr"].rolling(window=7, min_periods=1).mean()
city_hotel["adr_smooth"] = city_hotel["adr"].rolling(window=7, min_periods=1).mean()

plt.figure(figsize=(20, 8))

plt.plot(resort_hotel.index, resort_hotel["adr_smooth"], label="Resort Hotel", 
         color=sns.color_palette("rocket")[0], linewidth=3)
plt.plot(city_hotel.index, city_hotel["adr_smooth"], label="City Hotel", 
         color=sns.color_palette("rocket")[2], linewidth=3)

plt.ylabel("Average Daily Rate (€)", fontsize=16, fontweight="bold")

plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=30, fontsize=14)
plt.yticks(fontsize=14) 

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

plt.legend(fontsize=12, title="Hotel Type", title_fontsize=14)

plt.grid(alpha=0.3)

plt.show()


#############################
###### VIZUALIZATION 4 ######
#############################

booking_viz5 = booking.copy()

booking_viz5 = booking_viz5[booking_viz5["is_canceled"] == 1]

booking_viz5["days_until_cancellation"] = booking_viz5["lead_time"]

booking_viz5 = booking_viz5[booking_viz5["days_until_cancellation"] >= 0]

adr_bins = np.arange(0, booking_viz5["adr"].max() + 15, 20)  
booking_viz5["adr_bin"] = pd.cut(booking_viz5["adr"], bins=adr_bins, right=False)

booking_viz5 = booking_viz5.dropna(subset=["adr_bin"])

percentage_threshold = 0.01 
min_count = max(30, int(percentage_threshold * len(booking_viz5)))

bin_counts = booking_viz5["adr_bin"].value_counts()

adaptive_threshold = max(30, int(0.5 * bin_counts.median())) 

final_min_count = max(min_count, adaptive_threshold)

valid_bins = bin_counts[bin_counts >= final_min_count].index

# Filter dataset to only include valid bins
booking_viz5 = booking_viz5[booking_viz5["adr_bin"].isin(valid_bins)]

booking_viz5["adr_bin"] = booking_viz5["adr_bin"].apply(lambda x: f"{int(x.left)}-{int(x.right)}")

sorted_bins = sorted(booking_viz5["adr_bin"].unique(), key=lambda x: int(x.split("-")[0]))

fig, ax1 = plt.subplots(figsize=(12, 6))
kmf = KaplanMeierFitter()

time_points = np.arange(0, booking_viz5["days_until_cancellation"].max(), 5) 
percent_remaining = np.zeros_like(time_points, dtype=float)

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

ax2 = ax1.twinx()
ax2.plot(time_points, percent_remaining / 100, color="black", linestyle="dashed", linewidth=2, label="Data Remaining (%)")  # Normalize
ax2.set_ylabel("Percentage of Data Remaining", fontsize=22, fontweight="bold", color="black")

ax1.set_ylim(0, 1)  # Survival probability from 0 to 1
ax2.set_ylim(0, 1)  # Convert percentage to 0-1 scale (so 100% is  then 1.0)

ax1.set_xlabel("Days Since Booking (Lead Time)", fontsize=22, fontweight="bold")
ax1.set_ylabel("Survival Probability (Not Canceled)", fontsize=22, fontweight="bold")

ax1.tick_params(axis='x', labelsize=16)
ax1.tick_params(axis='y', labelsize=16)
ax2.tick_params(axis='y', labelsize=16)

handles, labels = ax1.get_legend_handles_labels()
sorted_legend = sorted(zip(labels, handles), key=lambda x: int(x[0].split("-")[0]))  # Sort numerically
labels, handles = zip(*sorted_legend)  # Unpack sorted tuples

ax1.legend(handles, labels, title="ADR (€)", fontsize=14, title_fontsize=18)  # Doubled font size
ax2.legend(loc="center right", fontsize=14, title_fontsize=15)  # Doubled font size

ax1.grid(True, linestyle="--", alpha=0.5)
ax2.grid(False)  # Disable separate grid for the secondary y-axis

plt.xticks(rotation=45)

plt.show()

#############################
###### VIZUALIZATION 5 ######
#############################

sns.set_style("whitegrid")

# Create a new variable: Time until cancellation (only for canceled bookings)
booking["leadtime_of_cancelled_bookings"] = booking["lead_time"].where(booking["is_canceled"] == 1, np.nan)

test = booking.copy()

# Select only numerical columns, ensuring 'leadtime_of_cancelled_bookingsn' and 'adr' are included
numeric_test = test.select_dtypes(include=['number'])

plt.figure(figsize=(20, 10))
sns.heatmap(numeric_test.corr(), annot=True, linecolor='black', linewidths=2, cmap='rocket', fmt=".2f")

plt.subplots_adjust(left=0.2, bottom=0.2)
plt.xticks(rotation=45, ha='right')

plt.show()

#############################
###### VIZUALIZATION 6 ######
#############################

# Convert reservation status date to datetime if applicable
if "reservation_status_date" in booking.columns:
    booking["reservation_status_date"] = pd.to_datetime(booking["reservation_status_date"])

# Extract month names from reservation status date
booking["month"] = booking["reservation_status_date"].dt.month_name()

# Count cancellations per month
monthly_cancellations = booking[booking["is_canceled"] == 1]["month"].value_counts().reindex(
    ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
)

sns.set_style("whitegrid")

# Create bar plot
plt.figure(figsize=(12, 6))
ax = sns.barplot(x=monthly_cancellations.index, y=monthly_cancellations.values, 
                 palette="rocket", edgecolor="black")

# labels on top of bars
for p in ax.patches:
    ax.annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=12, fontweight="bold", color="black")

# Format y-axis in thousands (K)
ax.get_yaxis().set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))

plt.xlabel("Month", fontsize=14, fontweight="bold")
plt.ylabel("Number of Cancellations", fontsize=14, fontweight="bold")
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

sns.despine()

plt.grid(alpha=0)

plt.show()

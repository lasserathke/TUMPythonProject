import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D


df = pd.read_csv(r"cleaned_dataset.csv")
columns_not_needed = ['name', 'email', 'phone-number', 'credit_card',
                      'reservation_status', 'reservation_status_date',
                      'arrival_date_month', 'hotel']
df_cleaned = df.drop(columns=[col for col in columns_not_needed if col in df.columns], errors='ignore')



# clustering based on 5 features (chose logically, could have implemented some ML stuff, but i think my computer would explode)
features = ['lead_time', 'adr', 'total_of_special_requests', 'required_car_parking_spaces', 'booking_changes']
df_cluster_extended = df_cleaned[features].copy()

scaler = StandardScaler()
df_cluster_scaled_extended = scaler.fit_transform(df_cluster_extended)

# DBSCAN !!
dbscan_extended = DBSCAN(eps=1.2, min_samples=10)
clusters_extended = dbscan_extended.fit_predict(df_cluster_scaled_extended)
df_cleaned['Cluster_Extended'] = clusters_extended

# work in progress
sns.pairplot(df_cleaned, hue="clusters", vars=features, palette="viridis")
plt.show()

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df_cleaned['lead_time'], df_cleaned['adr'], df_cleaned['total_of_special_requests'],
                      c=df_cleaned['clusters'], cmap='viridis', alpha=0.7)
ax.set_xlabel('Lead Time')
ax.set_ylabel('ADR')
ax.set_zlabel('Total Special Requests')
plt.colorbar(scatter, label="clusters")
plt.title("3D DBSCAN clustering")
plt.show()

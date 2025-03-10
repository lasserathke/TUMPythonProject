import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan
from sklearn.preprocessing import StandardScaler


# HDBSCAN - clustering based on density


df = pd.read_csv("cleaned_dataset.csv")
columns_not_needed = ['name', 'email', 'phone-number', 'credit_card',
                      'reservation_status', 'reservation_status_date',
                      'arrival_date_month', "hotel"]
df_cleaned = df.drop(columns=[col for col in columns_not_needed if col in df.columns], errors='ignore')

# choice of features for clustering - we wanted to focus on revenue so we chose the 'lead_time' and 'adr' columns
features_L_A = ['lead_time', 'adr']
df_cluster_L_A = df_cleaned[features_L_A].copy()

scaler = StandardScaler()
df_cluster_scaled_1 = scaler.fit_transform(df_cluster_L_A)

# HDBSCAN
hdbscan_1 = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=5)
df_cleaned['Cluster'] = hdbscan_1.fit_predict(df_cluster_scaled_1)


# plottin' v1
plt.figure(figsize=(10, 5))
sns.scatterplot(x=df_cleaned['lead_time'], y=df_cleaned['adr'], hue=df_cleaned['Cluster'], palette='viridis', alpha=0.75)
plt.xlabel('Lead Time')
plt.ylabel('ADR')
plt.title('HDBSCAN Clustering: Lead Time vs ADR')
plt.legend(title='Cluster')
plt.show()

# clustering using different features
features_A_R = ['adr', 'total_of_special_requests']
df_cluster_A_R = df_cleaned[features_A_R].copy()
df_cluster_scaled_2 = scaler.fit_transform(df_cluster_A_R)

hdbscan_2 = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=5)
df_cleaned['Cluster_A_R'] = hdbscan_2.fit_predict(df_cluster_scaled_2)

# plottin' v2
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_cleaned['adr'], y=df_cleaned['total_of_special_requests'], hue=df_cleaned['Cluster_A_R'], palette='viridis', alpha=0.75)
plt.xlabel('ADR')
plt.ylabel('Total Special Requests')
plt.title('HDBSCAN Clustering: ADR vs Special Requests')
plt.legend(title='Cluster')
plt.show()

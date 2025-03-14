import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan
from sklearn.preprocessing import StandardScaler
import time
from sklearn.metrics import silhouette_score
from collections import Counter



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


# plottin'
plt.figure(figsize=(10, 5))
sns.scatterplot(x=df_cleaned['lead_time'], y=df_cleaned['adr'], hue=df_cleaned['Cluster'], palette='viridis', alpha=0.75)
plt.xlabel('Lead Time')
plt.ylabel('ADR')
plt.title('HDBSCAN Clustering: Lead Time vs ADR')
plt.legend(title='Cluster')
plt.show()



def evaluate_clustering(model, data, labels):
    unique_clusters = set(labels)
    silhouette = silhouette_score(data, labels) if len(unique_clusters) > 1 and -1 not in labels else 'N/A'

    log_likelihood = "N/A"

    cluster_counts = dict(Counter(labels))
    total_points = len(labels)
    
    cluster_sizes_percent = {k: (v / total_points) * 100 for k, v in cluster_counts.items()}
    
    plt.figure(figsize=(8, 8))
    plt.pie(cluster_sizes_percent.values(), colors=plt.cm.Paired.colors, startangle=90, wedgeprops={'edgecolor': 'none'})
    plt.title('Cluster Size Distribution (%)')
    plt.axis('equal')
    plt.show()


    start_time = time.time()
    model.fit(data)
    fit_time = time.time() - start_time

    result = f"""
    Fit Time (seconds): {fit_time:.4f}
    Log-Likelihood: {log_likelihood}
    Silhouette Score: {silhouette}
    Cluster Size Distribution (%): {cluster_sizes_percent}
    """
    
    return result

hdbscan_kpis = evaluate_clustering(hdbscan_1, df_cluster_scaled_1, df_cleaned['Cluster'])
print("HDBSCAN KPIs:\n", hdbscan_kpis)

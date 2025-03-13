import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
from scipy.spatial import Voronoi, voronoi_plot_2d
import seaborn as sns
import time
from collections import Counter


df = pd.read_csv (r"cleaned_dataset.csv")



df.describe()

columns_not_needed = ['name', 'email', 'phone-number', 'credit_card', 
                   'reservation_status', 'reservation_status_date', 
                   'arrival_date_month', 'hotel']

df_cleaned = df.drop(columns=[col for col in columns_not_needed if col in df.columns], errors='ignore')


features_L_A = ['lead_time', 'adr']  #The features that we will cluster
df_cluster_L_A = df_cleaned[features_L_A].copy()


scaler = StandardScaler()
df_cluster_scaled_1 = scaler.fit_transform(df_cluster_L_A)

wcss = []
K_range = range(2, 11) # K should be more than 2

for k in K_range:
    kmeans_1 = KMeans(n_clusters=k, random_state=40, n_init=10)
    kmeans_1.fit(df_cluster_scaled_1)
    wcss.append(kmeans_1.inertia_)


plt.figure(figsize=(10, 5))
plt.plot(K_range, wcss, marker='o', linestyle='-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Within-Cluster Sum of Squares')
plt.title('Elbow Method')
plt.show()

chosen_k_1 = 4  #Biggest drop, less blending 
kmeans_1 = KMeans(n_clusters=chosen_k_1, random_state=40, n_init=10)
df_cleaned_1 = df_cleaned.loc[df_cluster_L_A.index]  # Ensure alignment
df_cleaned_1['Cluster'] = kmeans_1.fit_predict(df_cluster_scaled_1)


plt.figure(figsize=(10, 5))
scatter = plt.scatter(df_cleaned_1['lead_time'], df_cleaned_1['adr'], c=df_cleaned_1['Cluster'], cmap='viridis', alpha=0.75)
plt.xlabel('Lead Time')
plt.ylabel('ADR')
plt.title(f'K-Means Clustering, Lead Time vs ADR')


centers = scaler.inverse_transform(kmeans_1.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centroids')


centroid_legend = mpatches.Patch(color='red', label='Centroids (X)')
plt.legend(handles=[centroid_legend], loc='upper right')


plt.show()


features_A_R = ['adr', 'total_of_special_requests']
df_cluster_A_R = df_cleaned[features_A_R].copy()
scaler = StandardScaler()
df_cluster_scaled_2 = scaler.fit_transform(df_cluster_A_R)

wcss = []
k_range = range(1, 11)
for k in k_range:
    kmeans_2 = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_2.fit(df_cluster_scaled_2)
    wcss.append(kmeans_2.inertia_)

#Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(k_range, wcss, marker='o', linestyle='-')
plt.xlabel('Number of Clusters Needed (K)')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal K')
plt.grid()
plt.show()

chosen_k_2 = 3  
kmeans = KMeans(n_clusters= chosen_k_2, random_state=40, n_init=10)
df['Cluster'] = kmeans_2.fit_predict(df_cluster_scaled_2)


plt.figure(figsize=(10, 6))
colors = ['blue', 'green', 'orange']  
for i in range(chosen_k_2):
    cluster_points_2 = df[df['Cluster'] == i]
    plt.scatter(cluster_points_2['adr'], cluster_points_2['total_of_special_requests'],
                color=colors[i], label=f'Cluster {i+1}', alpha=0.5)


centers = scaler.inverse_transform(kmeans_2.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='o', s=200, label='Centroid')


plt.xlabel('ADR')
plt.ylabel('Total Special Requests')
plt.title(f'K-Means Clustering, ADR vs Special Requests')
plt.legend()

plt.show()

kmeans_clusters = kmeans_2.fit_predict(df_cluster_scaled_2)


def evaluate_clustering(model, data, labels):

    start_time = time.time()
    model.fit(data)
    fit_time = time.time() - start_time
    
    unique_clusters = set(labels)
    
    silhouette = silhouette_score(data, labels) if len(unique_clusters) > 1 and -1 not in labels else 'N/A'
    log_likelihood = model.score(data) if hasattr(model, "score") else None
    
    noise_percentage = (np.sum(labels == -1) / len(labels)) * 100 if -1 in labels else 0
    
    cluster_counts = dict(Counter(labels))
    total_points = len(labels)
    cluster_sizes_percent = {k: (v / total_points) * 100 for k, v in cluster_counts.items()}
    
    plt.figure(figsize=(8, 8))
    plt.pie(cluster_sizes_percent.values(), colors=plt.cm.Paired.colors, startangle=90, wedgeprops={'edgecolor': 'none'})
    plt.title('Cluster Size Distribution (%)')
    plt.axis('equal')
    plt.show()
    
    
    
    result = f"""
    Fit Time (seconds): {fit_time:.4f}
    Log-Likelihood: {log_likelihood if log_likelihood is not None else 'N/A'}
    Silhouette Score: {silhouette} 
    Cluster Size Distribution (%): {cluster_sizes_percent}
    """
    
    return result

kmeans_kpis = evaluate_clustering(kmeans_1, df_cluster_scaled_1, df_cleaned_1['Cluster'])
print("KMeans KPIs:\n", kmeans_kpis)

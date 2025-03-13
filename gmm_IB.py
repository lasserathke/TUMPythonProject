from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator
import time
from sklearn.metrics import silhouette_score
from collections import Counter


# GAUSSIAN MIXTURE MODEL - improved k-means


df = pd.read_csv("cleaned_dataset.csv")
columns_not_needed = ['name', 'email', 'phone-number', 'credit_card',
                      'reservation_status', 'reservation_status_date',
                      'arrival_date_month', 'hotel']
df_cleaned = df.drop(columns=[col for col in columns_not_needed if col in df.columns], errors='ignore')

# choice of features for clustering - we wanted to focus on revenue so we chose the 'lead_time' and 'adr' columns
features_L_A = ['lead_time', 'adr']
df_cluster_L_A = df_cleaned[features_L_A].copy()

scaler = StandardScaler()
df_cluster_scaled_1 = scaler.fit_transform(df_cluster_L_A)

# GMM
# BIC method
bic_scores = []
n_range = range(1, 10)
log_likelihood_scores = []

for n in n_range:
    gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
    gmm.fit(df_cluster_scaled_1)

    bic = gmm.bic(df_cluster_scaled_1)
    bic_scores.append(bic)

    aic = gmm.aic(df_cluster_scaled_1)
    aic_scores.append(aic)
    log_likelihood_scores.append(gmm.score(df_cluster_scaled_1))
    
    print("n:", n, "BIC:", bic, "AIC:", aic)

# plottin' BIC
plt.figure(figsize=(10, 5))
plt.plot(n_range, bic_scores, label='BIC')
plt.xlabel("Number of clusters")
plt.ylabel("BIC")
plt.legend()
plt.title("Number of clusters vs BIC in GMM")
plt.show()

n_components = n_range[np.argmin(bic_scores)]
#print(f"Optimal number of clusters (based on BIC): {n_components}") says 9, because BIC is smallest, but we chose the knee
n_components = 4

# model
gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42, n_init=10)
df_cleaned['Cluster'] = gmm.fit_predict(df_cluster_scaled_1)

# plottin' clusters
plt.figure(figsize=(10, 5))
sns.scatterplot(x=df_cleaned['lead_time'], y=df_cleaned['adr'], hue=df_cleaned['Cluster'], palette='viridis', alpha=0.75)
plt.xlabel('Lead Time')
plt.ylabel('ADR')
plt.title('GMM Clustering: Lead Time vs ADR')
plt.legend(title='Cluster')
plt.show()



def evaluate_clustering(model, data, labels):
    start_time = time.time()
    model.fit(data)
    fit_time = time.time() - start_time
    
    silhouette = silhouette_score(data, labels) if -1 not in labels else None
    log_likelihood = model.score(data) if hasattr(model, "score") else None
    
    
    cluster_counts = dict(Counter(labels))
    total_points = len(labels)
    cluster_sizes_percent = {k: (v / total_points) * 100 for k, v in cluster_counts.items()}
    
    plt.figure(figsize=(8, 8))
    plt.pie(cluster_sizes_percent.values(), colors=plt.cm.Paired.colors, startangle=90, wedgeprops={'edgecolor': 'none'})
    plt.title('Cluster Size Distribution (%)')
    plt.axis('equal')
    plt.show()
    
    
    result = f"""
    Fit Time (s): {fit_time:.4f}
    Log-Likelihood: {log_likelihood if log_likelihood is not None else 'N/A'}
    Silhouette Score: {silhouette if silhouette is not None else 'N/A'}
    Cluster Size Distribution (%): {cluster_sizes_percent}
    """
    
    
    return result

gmm_kpis = evaluate_clustering(gmm, df_cluster_scaled_1, df_cleaned['Cluster'])
print("GMM KPIs:\n", gmm_kpis)

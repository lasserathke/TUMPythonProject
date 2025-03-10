from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator


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
# choice of cluster number using BIC method
bic_scores = []
aic_scores = []
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
plt.plot(n_range, bic_scores, marker='o', label='BIC')
plt.plot(n_range, aic_scores, marker='s', label='AIC')
plt.xlabel("# of clusters")
plt.ylabel("BIC / AIC")
plt.legend()
plt.title("# of clusters vs BIC & AIC in GMM")
plt.show()

n_components = n_range[np.argmin(bic_scores)]
print(f"Optimal number of clusters (based on BIC): {n_components}")

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


# clustering using different features
features_A_R = ['adr', 'total_of_special_requests']
df_cluster_A_R = df_cleaned[features_A_R].copy()
df_cluster_scaled_2 = scaler.fit_transform(df_cluster_A_R)

gmm_2 = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42, n_init=10)
df_cleaned['Cluster_A_R'] = gmm_2.fit_predict(df_cluster_scaled_2)

# plottin' v2
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_cleaned['adr'], y=df_cleaned['total_of_special_requests'], hue=df_cleaned['Cluster_A_R'], palette='viridis', alpha=0.75)
plt.xlabel('ADR')
plt.ylabel('Total Special Requests')
plt.title('GMM Clustering: ADR vs Special Requests')
plt.legend(title='Cluster')
plt.show()


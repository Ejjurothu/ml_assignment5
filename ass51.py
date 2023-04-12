# importing the necessary libraries
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# loading the CC dataset
cc_df = pd.read_csv('CC.csv')

# checking for missing values
print(cc_df.isnull().sum())

# dropping the rows with missing values
cc_df.dropna(inplace=True)

# dropping the 'CUST_ID' column as it is not useful for analysis
cc_df.drop('CUST_ID', axis=1, inplace=True)

# a. applying PCA
pca = PCA(n_components=2)
cc_df_pca = pca.fit_transform(cc_df)

# creating a new dataframe with the PCA components
cc_df_pca_df = pd.DataFrame(data=cc_df_pca, columns=['PC1', 'PC2'])

# b. applying k-means on the PCA components
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(cc_df_pca_df)

# calculating the silhouette score
silhouette_score_pca = silhouette_score(cc_df_pca_df, kmeans.labels_)
print(f"Silhouette score after PCA: {silhouette_score_pca}")

# c. scaling the dataset, applying PCA, and applying k-means on the PCA components
scaler = StandardScaler()
cc_df_scaled = scaler.fit_transform(cc_df)

pca = PCA(n_components=2)
cc_df_pca_scaled = pca.fit_transform(cc_df_scaled)

kmeans_scaled = KMeans(n_clusters=2, random_state=42)
kmeans_scaled.fit(cc_df_pca_scaled)

silhouette_score_scaled_pca = silhouette_score(cc_df_pca_scaled, kmeans_scaled.labels_)
print(f"Silhouette score after Scaling+PCA+K-Means: {silhouette_score_scaled_pca}")

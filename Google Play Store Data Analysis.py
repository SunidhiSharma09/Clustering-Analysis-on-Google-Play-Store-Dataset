#!/usr/bin/env python
# coding: utf-8

# ### Finding the best fit clustering technique to analyse Google Play Store Dataset

# This  project  focuses  on  performing  a  clustering  analysis  on  the Google  Play  Store  dataset  to  identify  patterns  and  group  similar applications  together.  The  goal  is  to  determine  which clustering technique best fits the data set by comparing different algorithms and then analyse playstore data using the particular technique.

# In[ ]:





# In[51]:


# Overview of the data.

import pandas as pd 
data= pd.read_csv(r"C:\Users\DELL\Documents\MACHINE LEARNING\googleplaystore.csv")
data.head()


# # Checking the shape of the data before applying clustering techniques by plotting 3D and pair plot.

# In[7]:


# Plotting 3-D scatter plot of features (Install, Size, Price and Rating) to get an idea of the data shape.

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Loading the data from your CSV file
df = pd.read_csv( r'C:\Users\DELL\Documents\MACHINE LEARNING\googleplaystore.csv')

# Removing NaN values from 'Installs' and converting to numeric form
df = df[df['Installs'].str.contains(r'\d', na=False)]
df['Installs'] = df['Installs'].str.replace('[+,]', '', regex=True).astype(int)

# function to convert 'Size' to numeric form
def size_to_numeric(size):
    if 'M' in size:
        return float(size.replace('M', '')) * 1e6
    elif 'K' in size:
        return float(size.replace('K', '')) * 1e3
    return None  # Handle 'Varies with device' or NaN cases

df['Size'] = df['Size'].apply(size_to_numeric)

# Converting 'Price' to numeric form
df['Price'] = df['Price'].str.replace('$', '').replace('Free', '0').astype(float)

# Dropping NaN values rows
df = df.dropna(subset=['Rating', 'Size', 'Installs', 'Price'])

# Plotting a 3D scatter graph with Rating, Size, and Installs
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df['Rating'], df['Size'], df['Installs'], c=df['Price'], cmap='viridis', s=50)
ax.set_xlabel('Rating')
ax.set_ylabel('Size (Bytes)')
ax.set_zlabel('Installs')
cbar = fig.colorbar(scatter)
cbar.set_label('Price')
plt.title('3D Scatter Plot of Google Play Store Apps Data (Rating, Size, Installs and Price)')
plt.show()


# In[6]:


# Plotting a pair plot of various numeric features of Google play store data. (to get the shape of idea in 2-D form also)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Loading the data
df = pd.read_csv(r"C:\Users\DELL\Documents\MACHINE LEARNING\googleplaystore.csv")

# Converting columns to numeric form
df['Price'] = df['Price'].replace('[\$,]', '', regex=True)
df['Size'] = df['Size'].replace('[\D]', '', regex=True)
df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')
df['Installs'] = pd.to_numeric(df['Installs'].replace('[\+,]', '', regex=True), errors='coerce')

df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df['Size'] = pd.to_numeric(df['Size'], errors='coerce')

# Dropping NaN value rows
df_numeric = df[['Rating', 'Reviews', 'Size', 'Installs', 'Price']].dropna()

# Ensuring df_numeric is not empty before proceeding
if df_numeric.empty:
    raise ValueError("The DataFrame after cleaning has no data. Check the data cleaning steps.")

# PLotting pair plot
sns.pairplot(df_numeric)
plt.suptitle("Pair Plot of Google Play Store Data - Rating, Review, Size, Install & Price", y=1.02)
plt.show()


# In[ ]:





# In[ ]:





# # K-MEANS CLUSTERING TECHNIQUE

# In[ ]:





# In[2]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Loading the dataset from a CSV file
df = pd.read_csv(r"C:\Users\DELL\Documents\MACHINE LEARNING\googleplaystore.csv")

# Dropping NaN value rows
df.dropna(inplace=True)

# Function to convert 'Size' from strings to numerical values in MB
def convert_size(size):
    if isinstance(size, str):
        if 'M' in size:
            return float(size.replace('M', ''))
        elif 'k' in size:
            return float(size.replace('k', '')) / 1024  # Convert KB to MB
    return np.nan

# Function to convert 'Installs' from strings to numerical values
def convert_installs(installs):
    if isinstance(installs, str):
        try:
            return int(installs.replace(',', '').replace('+', ''))
        except ValueError:
            return np.nan
    return np.nan

# Function to convert 'Price' from strings to numerical values
def convert_price(price):
    if isinstance(price, str):
        try:
            return float(price.replace('$', '')) if price != '0' else 0.0
        except ValueError:
            return np.nan
    return np.nan

# conversions
df['Size'] = df['Size'].apply(convert_size)
df['Installs'] = df['Installs'].apply(convert_installs)
df['Price'] = df['Price'].apply(convert_price)

# Handling NaN values by filling them with column means
df.fillna(df.mean(numeric_only=True), inplace=True)

# features selection
numerical_features = df[['Rating', 'Reviews', 'Size', 'Installs', 'Price']]

# Standardizing the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(numerical_features)

# Using Silhouette Score method to determine the optimal number of clusters
silhouette_avg = []
kmeans_models = []

for n_clusters in range(2, 11):  # Silhouette score requires at least 2 clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_features)
    silhouette_avg.append(silhouette_score(scaled_features, cluster_labels))
    kmeans_models.append(kmeans)

# Plotting Silhouette Scores
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_avg, marker='o', linestyle='-')
plt.title('Silhouette Scores For Optimal Number of Clusters')
print("")
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

print('')

# optimal number of clusters based on the highest Silhouette Score
optimal_n_clusters = range(2, 11)[silhouette_avg.index(max(silhouette_avg))]
print(f"Optimal number of clusters: {optimal_n_clusters}")

# building K-means model 

# Training the K-Means model
kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42)
clusters = kmeans.fit_predict(scaled_features)

print('')

# # Getting the cluster centers and centroids
cluster_centers = kmeans.cluster_centers_

print('')
# original scale of the cluster centers
original_centers = scaler.inverse_transform(cluster_centers)

# Adding cluster to the original DataFrame
df['Cluster'] = clusters

print('')
# printing DataFrame withcluster
print(df.head())

print('')

# final cluster centers and centroids
print(f"Final cluster centers (in original scale): \n{original_centers}")
print('')
print(f"Final cluster centroids (in standardized scale): \n{cluster_centers}")


# In[4]:


# plotting cluster for rating and installs features

import seaborn as sns
plt.figure(figsize=(10, 4))
sns.scatterplot(x=df['Rating'], y=df['Installs'], hue=df['Cluster'], palette='viridis', s=100, alpha=0.7)
plt.title('Apps clustering based on rating and installs')
plt.show()


# In[6]:


# plotting cluster for price and installs features

import seaborn as sns
plt.figure(figsize=(10, 4))
sns.scatterplot(x=df['Price'], y=df['Installs'], hue=df['Cluster'], palette='viridis', s=100, alpha=0.7)
plt.title('Apps clustering based on price and installs')
plt.show()


# In[7]:


# plotting cluster for size and installs features

import seaborn as sns
plt.figure(figsize=(10, 4))
sns.scatterplot(x=df['Size'], y=df['Installs'], hue=df['Cluster'], palette='viridis', s=100, alpha=0.7)
plt.title('Apps clustering based on size and installs')
plt.show()


# In[ ]:





# In[ ]:





# # GMM CLUSTERING TECHNIQUE

# In[ ]:





# In[1]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt

# Loading the data from CSV
df = pd.read_csv(r"C:\Users\DELL\Documents\MACHINE LEARNING\googleplaystore.csv")

# function to convert 'Size' to numeric form
def convert_size(size_str):
    if 'M' in size_str:
        return float(size_str.replace('M', '')) * 1e6
    elif 'k' in size_str:
        return float(size_str.replace('k', '')) * 1e3
    return np.nan

df['Size'] = df['Size'].apply(convert_size)

# Converting 'Installs' from string to numeric form
df['Installs'] = df['Installs'].str.replace('+', '').str.replace(',', '')

# Handle non-numeric values in installs
df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce')
df = df.dropna(subset=['Installs'])

# function to convert 'Price' to numeric form
def convert_price(price_str):
    return float(price_str.replace('$', '')) if price_str != '0' else 0.0

df['Price'] = df['Price'].apply(convert_price)

# Selecting the features for clustering and dropping NaN rows
X = df[['Rating', 'Reviews', 'Size', 'Installs', 'Price']].dropna()

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Function to calculate AIC and BIC to find optimal number of clusters.
def calculate_aic_bic(X, max_clusters=10):
    aic = []
    bic = []
    for n_clusters in range(1, max_clusters + 1):
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm.fit(X)
        aic.append(gmm.aic(X))
        bic.append(gmm.bic(X))
    
    return aic, bic

aic, bic = calculate_aic_bic(X_scaled, max_clusters=10)

# Plotting AIC and BIC scores
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), aic, label='AIC', marker='o')
plt.plot(range(1, 11), bic, label='BIC', marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('AIC and BIC for GMM')
plt.legend()
plt.show()

# optimal number of clusters based on the lowest AIC and BIC
optimal_n_clusters_aic = np.argmin(aic) + 1
optimal_n_clusters_bic = np.argmin(bic) + 1

print(f"Optimal number of clusters based on AIC: {optimal_n_clusters_aic}")
print(f"Optimal number of clusters based on BIC: {optimal_n_clusters_bic}")


# In[2]:


# optimal number of clusters based on lowest score 
optimal_n_clusters = 10   

# Creating GMM Model 
gmm = GaussianMixture(n_components=optimal_n_clusters, random_state=42)
df['Cluster'] = np.nan  # Initialize the Cluster column
cluster_labels = gmm.fit_predict(X_scaled)

# adding cluster to dataframe
df.loc[X.index, 'Cluster'] = cluster_labels

# Remove rows with NaN in the 'Cluster' column
df = df.dropna(subset=['Cluster'])

# printing the dataframe with cluster columns
print(df.head())


# In[3]:


# checking number of unique clusters.

np.unique(df['Cluster'])


# In[11]:


np.bincount(df['Cluster'])


# In[4]:


# Displaying means of each clusters

cluster_means = gmm.means_
print("\nCluster Means:")
print(cluster_means)


# In[5]:


# Displaying covariance of each cluster.

cluster_covariances = gmm.covariances_
print("\nCluster Covariances:")
print(cluster_covariances)


# In[6]:


# Displaying weights of each cluster.

Cluster_Weights = gmm.weights_
print("\nCluster Weights:")
print(Cluster_Weights)


# In[7]:


# Visualizing the clusters using two selected features -'Rating' and 'Installs'

plt.figure(figsize=(10, 4))
plt.scatter(df['Rating'], df['Installs'], c=cluster_labels, cmap='viridis', s=50, alpha=0.6)
plt.colorbar(label='Cluster')
plt.xlabel('Rating')
plt.ylabel('Installs')
plt.title('Cluster Visualization on Rating and Installs features')
plt.show()


# In[8]:


# Visualizing the clusters using two selected features -'Price' and 'Installs'

plt.figure(figsize=(10, 4))
plt.scatter(df['Price'], df['Installs'], c=cluster_labels, cmap='viridis', s=50, alpha=0.6)
plt.colorbar(label='Cluster')
plt.xlabel('Price')
plt.ylabel('Installs')
plt.title('Cluster Visualization on Price and Installs features')
plt.show()


# In[10]:


# Visualizing the clusters using two selected features -'Size' and 'Installs'

plt.figure(figsize=(15, 4))
plt.scatter(df['Size'], df['Installs'], c=cluster_labels, cmap='viridis', s=50, alpha=0.6)
plt.colorbar(label='Cluster')
plt.xlabel('Size')
plt.ylabel('Installs')
plt.title('Cluster Visualization on Size and Installs features')
plt.show()


# In[ ]:





# In[ ]:





# # HIERARCHICAL CLUSTERING TECHNIQUE 

# In[ ]:





# In[1]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import numpy as np
import matplotlib.pyplot as plt

# Loading the data from CSV
df = pd.read_csv(r"C:\Users\DELL\Documents\MACHINE LEARNING\googleplaystore.csv")

# Function to convert 'Size' to numeric form
def convert_size(size_str):
    if 'M' in size_str:
        return float(size_str.replace('M', '')) * 1e6
    elif 'k' in size_str:
        return float(size_str.replace('k', '')) * 1e3
    return np.nan

df['Size'] = df['Size'].apply(convert_size)

# Converting 'Installs' from string to numeric form
df['Installs'] = df['Installs'].str.replace('+', '').str.replace(',', '')

df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce')
df = df.dropna(subset=['Installs'])

# Function to convert 'Price' to numeric form
def convert_price(price_str):
    return float(price_str.replace('$', '')) if price_str != '0' else 0.0

df['Price'] = df['Price'].apply(convert_price)

# Selecting features for clustering and dropping NaN value rows
X = df[['Rating', 'Reviews', 'Size', 'Installs', 'Price']].dropna()

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[2]:


# plotting dendogram
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(20,10));

dend=shc.dendrogram(shc.linkage(X_scaled,method='ward'));

plt.axhline(y=130, color='red',linestyle='--');


# In[3]:


# developing hierarchical - agglomerative clustering model.

from sklearn.cluster import AgglomerativeClustering

X_scaled = pd.DataFrame(X_scaled).dropna()
X_scaled = X_scaled.values

clust = AgglomerativeClustering(n_clusters=2, linkage='ward')
clusters = clust.fit_predict(X_scaled)


# In[4]:


clusters


# In[5]:


# checking the unique clusters 

unique_clusters = np.unique(clusters)
print(f"Unique clusters: {unique_clusters}")


# In[6]:


# Adding clusters to the original DataFrame
df.loc[X.index, 'Cluster'] = clusters

# printing the resulting DataFrame with cluster labels
print(df.head())


# In[10]:


# plotting the App cluster for Installs and Rating

import seaborn as sns

plt.figure(figsize=(10,4));
sns.scatterplot(x=df['Rating'],y=df['Installs'],hue=df['Cluster']);
plt.title('Apps clustering based on Rating and Installs')


# In[9]:


# plotting the App cluster for Installs and Price

import seaborn as sns

plt.figure(figsize=(10,4));
sns.scatterplot(x=df['Price'],y=df['Installs'],hue=df['Cluster']);
plt.title('Apps clustering based on Price and Installs')


# In[8]:


# plotting the App cluster for Installs and Size

import seaborn as sns
plt.figure(figsize=(10,4));
sns.scatterplot(x=df['Size'],y=df['Installs'],hue=df['Cluster']);
plt.title('Apps clustering based on Size and Installs')


# In[ ]:





# In[13]:


# plotting the App cluster for Price and Rating

import seaborn as sns
plt.figure(figsize=(10,4));
sns.scatterplot(x=df['Rating'],y=df['Price'],hue=df['Cluster']);
plt.title('Apps clustering based on Price and Rating')


# In[ ]:





# In[15]:


# plotting the App cluster for Price and Reviews

import seaborn as sns
plt.figure(figsize=(10,4));
sns.scatterplot(x=df['Reviews'],y=df['Price'],hue=df['Cluster']);
plt.title('Apps clustering based on Price and Reviews')


# In[ ]:





# In[19]:


# plotting the App cluster for Size and rating

import seaborn as sns
plt.figure(figsize=(10,4));
sns.scatterplot(x=df['Rating'],y=df['Size'],hue=df['Cluster']);
plt.title('Apps clustering based on Size and Rating')


# In[20]:


# plotting the App cluster for Reviews and rating

import seaborn as sns
plt.figure(figsize=(10,4));
sns.scatterplot(x=df['Rating'],y=df['Reviews'],hue=df['Cluster']);
plt.title('Apps clustering based on Review and Rating')


# In[ ]:





# In[ ]:





# # DBSCAN CLUSTERING TECHNIQUE

# In[ ]:





# In[8]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

# Loading the data from CSV
df = pd.read_csv(r"C:\Users\DELL\Documents\MACHINE LEARNING\googleplaystore.csv")

# Function to convert 'Size' to numeric form
def convert_size(size_str):
    if 'M' in size_str:
        return float(size_str.replace('M', '')) * 1e6
    elif 'k' in size_str:
        return float(size_str.replace('k', '')) * 1e3
    return np.nan

df['Size'] = df['Size'].apply(convert_size)

# Converting 'Installs' from string to numeric form
df['Installs'] = df['Installs'].str.replace('+', '').str.replace(',', '')
df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce')
df = df.dropna(subset=['Installs'])

# function convert 'Price' to numeric form
def convert_price(price_str):
    return float(price_str.replace('$', '')) if price_str != '0' else 0.0

df['Price'] = df['Price'].apply(convert_price)

# Selecting features for clustering and dropping NaN value rows.
X = df[['Rating', 'Reviews', 'Size', 'Installs', 'Price']].dropna()

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[9]:


# k nearest neighbour method to find optimal epsilon value.

from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np

neighbors = NearestNeighbors(n_neighbors=5)   # as we have taken 5 features for clustering  
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)

# Sorting distances.
distances = np.sort(distances[:, 4], axis=0)  # use 4 because we used 5 neighbors

# plotting distance and data points.
plt.plot(distances)
plt.ylabel('Distance to 5th Nearest Neighbor')
plt.xlabel('Data Points Sorted by Distance')
plt.show()

so eps = 1 , considering elbow point# Now plotting graphs with eps = 1 and different min_samples values to find optimal number of clusters.
# In[10]:


# min_samples = 6 

dbscan=DBSCAN(eps=1,min_samples=6)
dbscan.fit(X_scaled)
set(dbscan.labels_)


# In[11]:


plt.figure(figsize=(8,4))
plt.scatter(X_scaled[:,0],X_scaled[:,1],c=dbscan.labels_);
plt.title('Graph for min_sample=6 and cluster=7')


# In[12]:


# min_samples = 7

dbscan=DBSCAN(eps=1,min_samples=7) # eps
dbscan.fit(X_scaled)
set(dbscan.labels_)


# In[13]:


plt.figure(figsize=(8,4))
plt.scatter(X_scaled[:,0],X_scaled[:,1],c=dbscan.labels_);
plt.title('Graph for min_sample=7 and cluster=5')


# In[14]:


# min_samples = 8

dbscan=DBSCAN(eps=1,min_samples=8) # eps
dbscan.fit(X_scaled)
set(dbscan.labels_)


# In[15]:


plt.figure(figsize=(8,4))
plt.scatter(X_scaled[:,0],X_scaled[:,1],c=dbscan.labels_);
plt.title('Graph for min_sample=8 and cluster=3')


# In[16]:


# min_samples = 11

dbscan=DBSCAN(eps=1,min_samples=11) # eps
dbscan.fit(X_scaled)
set(dbscan.labels_)


# In[17]:


plt.figure(figsize=(8,4))
plt.scatter(X_scaled[:,0],X_scaled[:,1],c=dbscan.labels_);
plt.title('Graph for min_sample=11 and cluster=3')


# In[18]:


# min_samples = 15  (any value from 8 to 14 giving 3 clusters only)

dbscan=DBSCAN(eps=1,min_samples=15) # eps
dbscan.fit(X_scaled)
set(dbscan.labels_)


# In[19]:


plt.figure(figsize=(8,4))
plt.scatter(X_scaled[:,0],X_scaled[:,1],c=dbscan.labels_);
plt.title('Graph for min_sample=15 and cluster=2')

Considering 2 as an optimal number of clusters in this technique.
# In[40]:


# creating DBSCAN model with 2 clusters.

dbscan = DBSCAN(eps=1, min_samples=15)
clusters = dbscan.fit_predict(X_scaled)

# Adding the cluster to the DataFrame
df_filtered = df.loc[X.index].copy()  # Filtering the original data frame to match the rows in X
df_filtered['Cluster'] = clusters

# Printing the resulting DataFrame with cluster labels
print(df_filtered[['App', 'Rating', 'Reviews', 'Size', 'Installs', 'Price', 'Cluster']].head())


# In[ ]:





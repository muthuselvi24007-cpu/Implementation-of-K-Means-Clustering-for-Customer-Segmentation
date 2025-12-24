# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Muthu selvi R
RegisterNumber: 25011064
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

# Step 2: Suppress warnings
warnings.filterwarnings("ignore")

# Step 3: Create a synthetic customer dataset
data = {
    'CustomerID': range(1, 21),
    'AnnualIncome': [15, 16, 17, 18, 19, 20, 60, 62, 64, 65, 66, 67,
                     120, 122, 124, 125, 126, 127, 128, 130],
    'SpendingScore': [39, 81, 6, 77, 40, 76, 6, 94, 3, 72, 14, 99,
                      15, 77, 13, 79, 35, 66, 29, 98]
}

df = pd.DataFrame(data)

# Step 4: Select features and scale them
X = df[['AnnualIncome', 'SpendingScore']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Step 6: Visualize the clusters
plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue']

for i in range(3):
    plt.scatter(
        df[df['Cluster'] == i]['AnnualIncome'],
        df[df['Cluster'] == i]['SpendingScore'],
        label=f'Cluster {i}',
        color=colors[i]
    )

# Plot centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    s=200,
    c='yellow',
    label='Centroids',
    marker='X'
)

plt.title('Customer Segmentation using K-Means')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.grid(True)
plt.show()
```

## Output:
<img width="978" height="698" alt="image" src="https://github.com/user-attachments/assets/16637d1f-f4ee-4e4c-bd86-7e23d20a0d8e" />



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.

import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------
# STEP 1: Load the Mall Customer Dataset
# Download from: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
# Save as Mall_Customers.csv in the same folder
# -----------------------------------------------

df = pd.read_csv("Mall_Customers.csv")

# We use Annual Income and Spending Score as features
X = df[["Annual Income (k$)", "Spending Score (1-100)"]].values

# -----------------------------------------------
# STEP 2: Scale the features
# -----------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------------------
# STEP 3: Train KMeans++ Model
# init="k-means++" is KMeans++ (smarter centroid init)
# n_clusters=5 is standard for this dataset
# -----------------------------------------------

# init="k-means++"
# 🔹 What it means:
# Specifies how initial centroids are chosen.
# Options:
# "random" → Random initialization
# "k-means++" → Smart probabilistic initialization
# 🔹 Why K-Means++ is better:
# Spreads centroids far apart
# Reduces chances of poor clustering
# Faster convergence


# n_init=10
# 🔹 What it means:
# Number of times the algorithm will run with different centroid seeds.



# max_iter=300
# 🔹 What it means:
# Maximum number of iterations allowed for a single run.
# 🔹 What is one iteration?
# Assign points to nearest centroid
# Recompute centroids

model = KMeans(
    n_clusters=5,
    init="k-means++",   # <-- This is what makes it KMeans++
    n_init=10,
    max_iter=300,
    random_state=42
)
model.fit(X_scaled)

# -----------------------------------------------
# STEP 4: Save model and scaler
# -----------------------------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully!")
print(f"Number of clusters: {model.n_clusters}")
print(f"Cluster labels found: {np.unique(model.labels_)}")

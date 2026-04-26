import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("query.csv")

# Keep only useful columns
df = df[["time", "latitude", "longitude", "depth", "mag"]]

# Drop rows with missing latitude/longitude
df = df.dropna(subset=["latitude", "longitude"])

# Optional filter to reduce clutter
df = df[df["mag"] >= 3.0]

# -----------------------------
# Select features
# -----------------------------
X = df[["latitude", "longitude"]]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# K-means: try multiple K values
# -----------------------------
k_values = [2, 3, 4, 5, 6, 7, 8]
scores = {}

for k in k_values:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans_temp.fit_predict(X_scaled)
    scores[k] = silhouette_score(X_scaled, labels)

best_k = max(scores, key=scores.get)

print("K-means silhouette scores:", scores)
print("Best K:", best_k)

# Final K-means model
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["kmeans_cluster"] = kmeans.fit_predict(X_scaled)

# -----------------------------
# DBSCAN
# -----------------------------
dbscan = DBSCAN(eps=0.20, min_samples=5)
df["dbscan_cluster"] = dbscan.fit_predict(X_scaled)

# -----------------------------
# Metrics
# -----------------------------
kmeans_cluster_count = len(set(df["kmeans_cluster"]))
dbscan_cluster_count = len(set(df["dbscan_cluster"])) - (1 if -1 in df["dbscan_cluster"].values else 0)
dbscan_noise_count = (df["dbscan_cluster"] == -1).sum()

print("K-means clusters:", kmeans_cluster_count)
print("DBSCAN clusters:", dbscan_cluster_count)
print("DBSCAN noise points:", dbscan_noise_count)

kmeans_score = silhouette_score(X_scaled, df["kmeans_cluster"])
print("K-means silhouette score:", kmeans_score)

db_labels = df["dbscan_cluster"]
if len(set(db_labels)) > 1 and len(set(db_labels) - {-1}) > 1:
    dbscan_score = silhouette_score(X_scaled, db_labels)
    print("DBSCAN silhouette score:", dbscan_score)
else:
    print("DBSCAN silhouette score not valid for current parameters")

# -----------------------------
# Plot side by side
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

x_min, x_max = df["longitude"].min(), df["longitude"].max()
y_min, y_max = df["latitude"].min(), df["latitude"].max()

# -----------------------------
# K-means plot
# -----------------------------
axes[0].scatter(
    df["longitude"],
    df["latitude"],
    c=df["kmeans_cluster"],
    cmap="tab10",
    s=22,
    alpha=0.78
)

# Convert centroids back to original coordinates
centers_scaled = kmeans.cluster_centers_
centers_original = scaler.inverse_transform(centers_scaled)

# Orange centroid markers
axes[0].scatter(
    centers_original[:, 1],
    centers_original[:, 0],
    s=340,
    c="orange",
    edgecolors="black",
    linewidths=1.8,
    marker="o",
    label="Centroids",
    zorder=5
)

axes[0].scatter(
    centers_original[:, 1],
    centers_original[:, 0],
    s=75,
    c="white",
    edgecolors="black",
    linewidths=0.8,
    marker="o",
    zorder=6
)

for i, center in enumerate(centers_original):
    axes[0].text(
        center[1],
        center[0],
        f"C{i}",
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="center",
        color="black",
        zorder=7
    )

axes[0].set_title(f"K-means Clustering (K={best_k})", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Longitude", fontsize=11)
axes[0].set_ylabel("Latitude", fontsize=11)
axes[0].set_xlim(x_min, x_max)
axes[0].set_ylim(y_min, y_max)
axes[0].grid(True, linestyle="--", alpha=0.3)
axes[0].legend()

# -----------------------------
# DBSCAN plot
# -----------------------------
unique_labels = sorted(df["dbscan_cluster"].unique())
cluster_labels = [label for label in unique_labels if label != -1]
cluster_colors = plt.cm.Set2(np.linspace(0, 1, max(len(cluster_labels), 1)))

color_index = 0
region_num = 1

for label in unique_labels:
    cluster_points = df[df["dbscan_cluster"] == label]

    if label == -1:
        axes[1].scatter(
            cluster_points["longitude"],
            cluster_points["latitude"],
            c="lightgray",
            s=18,
            alpha=0.5,
            edgecolors="none",
            label="Scattered / Isolated Earthquakes"
        )
    else:
        axes[1].scatter(
            cluster_points["longitude"],
            cluster_points["latitude"],
            color=cluster_colors[color_index],
            s=24,
            alpha=0.82,
            edgecolors="black",
            linewidths=0.2,
            label=f"Dense Earthquake Region {region_num}"
        )

        center_lon = cluster_points["longitude"].mean()
        center_lat = cluster_points["latitude"].mean()

        axes[1].text(
            center_lon,
            center_lat,
            f"Region {region_num}",
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3", alpha=0.85)
        )

        color_index += 1
        region_num += 1

axes[1].set_title("DBSCAN Clustering", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Longitude", fontsize=11)
axes[1].set_ylabel("Latitude", fontsize=11)
axes[1].set_xlim(x_min, x_max)
axes[1].set_ylim(y_min, y_max)
axes[1].grid(True, linestyle="--", alpha=0.3)
axes[1].legend(fontsize=9)

plt.suptitle("Earthquake Clustering Comparison: K-means vs DBSCAN", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.show()
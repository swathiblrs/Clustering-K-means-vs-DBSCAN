"""
================================================================================
K-Means vs DBSCAN: Knoxville Amenities Clustering
================================================================================
CS 581 - Advanced Algorithms
Group Presentation

Dataset: Knoxville, Tennessee Amenities (OpenStreetMap API)
Features: Latitude & Longitude (2D spatial data)
Source: Real-world data from OpenStreetMap
================================================================================
"""

# =============================================================================
# 1. IMPORT LIBRARIES
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import warnings
import time
import os

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

print("=" * 70)
print("K-Means vs DBSCAN: Knoxville Amenities Clustering")
print("=" * 70)

# =============================================================================
# 2. FETCH DATA FROM OPENSTREETMAP
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 2: Loading Data from OpenStreetMap API")
print("=" * 70)

import osmnx as ox
import geopandas as gpd

place = "Knoxville, Tennessee, USA"
print(f"\nLocation: {place}")

print("Fetching area boundary...")
start_time = time.time()
area = ox.geocode_to_gdf(place)

tags = {"amenity": True}
print("Fetching amenities from OpenStreetMap API...")
gdf = ox.features_from_polygon(area.geometry.iloc[0], tags)
fetch_time = time.time() - start_time
print(f"✓ Data fetched in {fetch_time:.2f} seconds")

# Filter to Point geometries only
gdf = gdf[gdf.geometry.type == "Point"].copy()

# Remove unwanted amenities
gdf = gdf[~gdf["amenity"].isin(["street_lamp", "letter_box"])]

print(f"\n✓ Total amenities loaded: {len(gdf)}")

# =============================================================================
# 3. DATA EXPLORATION
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 3: Data Exploration")
print("=" * 70)

# Amenity counts
print("\n📊 TOP 20 AMENITY TYPES IN KNOXVILLE:")
print("-" * 50)
counts = gdf["amenity"].value_counts()
for i, (amenity, count) in enumerate(counts.head(20).items()):
    print(f"  {i+1:2}. {amenity:<25} {count:>4} points")

print(f"\n  Total unique amenity types: {len(counts)}")
print(f"  Total amenities: {len(gdf)}")

# Create DataFrame for analysis
df = pd.DataFrame({
    'amenity': gdf['amenity'].values,
    'name': gdf['name'].values if 'name' in gdf.columns else ['Unknown'] * len(gdf)
})

# =============================================================================
# 4. DATA PREPROCESSING
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 4: Data Preprocessing")
print("=" * 70)

# Convert to projected coordinate system (meters)
gdf_projected = gdf.to_crs(epsg=3857)

# Extract X, Y coordinates
X = np.vstack([
    gdf_projected.geometry.x,
    gdf_projected.geometry.y
]).T

print(f"\nFeatures: Latitude, Longitude (2D)")
print(f"Samples: {len(X)}")
print(f"Coordinate System: EPSG:3857 (Web Mercator - meters)")

# Scale data for better clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\n✓ Data scaled (mean=0, std=1)")

# Keep lat/lon for mapping
gdf_latlon = gdf.to_crs(epsg=4326)

# PCA for visualization (even though 2D, good practice)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA Variance Explained:")
print(f"  PC1: {pca.explained_variance_ratio_[0]*100:.1f}%")
print(f"  PC2: {pca.explained_variance_ratio_[1]*100:.1f}%")
print(f"  Total: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# =============================================================================
# 5. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 5: Exploratory Data Analysis (EDA)")
print("=" * 70)

# 5.1 Spatial distribution
print("\n5.1 Generating spatial distribution plot...")
fig, ax = plt.subplots(figsize=(12, 10))
scatter = ax.scatter(X[:, 0], X[:, 1], c='steelblue', s=20, alpha=0.6, edgecolors='black', linewidths=0.3)
ax.set_xlabel('X (meters)', fontsize=12)
ax.set_ylabel('Y (meters)', fontsize=12)
ax.set_title(f'Knoxville Amenities Distribution\n({len(X)} points)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('01_spatial_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved: 01_spatial_distribution.png")

# 5.2 Amenity type distribution (bar chart)
print("\n5.2 Generating amenity type distribution...")
fig, ax = plt.subplots(figsize=(14, 8))
top_amenities = counts.head(15)
colors = plt.cm.viridis(np.linspace(0, 0.8, len(top_amenities)))
bars = ax.barh(range(len(top_amenities)), top_amenities.values, color=colors, edgecolor='black')
ax.set_yticks(range(len(top_amenities)))
ax.set_yticklabels(top_amenities.index)
ax.invert_yaxis()
ax.set_xlabel('Count', fontsize=12)
ax.set_title('Top 15 Amenity Types in Knoxville', fontsize=14, fontweight='bold')

# Add count labels
for bar, count in zip(bars, top_amenities.values):
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, str(count), va='center', fontsize=10)

plt.tight_layout()
plt.savefig('02_amenity_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved: 02_amenity_distribution.png")

# 5.3 Density heatmap
print("\n5.3 Generating density heatmap...")
fig, ax = plt.subplots(figsize=(12, 10))
hb = ax.hexbin(X[:, 0], X[:, 1], gridsize=30, cmap='YlOrRd', mincnt=1)
cb = plt.colorbar(hb, ax=ax, label='Count')
ax.set_xlabel('X (meters)', fontsize=12)
ax.set_ylabel('Y (meters)', fontsize=12)
ax.set_title('Amenity Density Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('03_density_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved: 03_density_heatmap.png")

# 5.4 Coordinate statistics
print("\n5.4 Coordinate Statistics:")
coord_stats = pd.DataFrame({
    'X (meters)': X[:, 0],
    'Y (meters)': X[:, 1]
}).describe().round(2)
print(coord_stats)

# =============================================================================
# 6. K-MEANS CLUSTERING
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 6: K-Means Clustering")
print("=" * 70)

# 6.1 Finding Optimal K using multiple methods
print("\n6.1 Finding Optimal K...")

K_range = range(2, 16)
inertias = []
silhouette_scores_kmeans = []
calinski_scores = []
davies_scores = []

print(f"\n{'K':<5} {'Inertia':<15} {'Silhouette':<12} {'Calinski-H':<15} {'Davies-B':<12}")
print("-" * 65)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    inertias.append(kmeans.inertia_)
    sil = silhouette_score(X_scaled, labels)
    cal = calinski_harabasz_score(X_scaled, labels)
    dav = davies_bouldin_score(X_scaled, labels)
    
    silhouette_scores_kmeans.append(sil)
    calinski_scores.append(cal)
    davies_scores.append(dav)
    
    print(f"{k:<5} {kmeans.inertia_:<15.2f} {sil:<12.4f} {cal:<15.2f} {dav:<12.4f}")

optimal_k_silhouette = list(K_range)[np.argmax(silhouette_scores_kmeans)]
optimal_k_calinski = list(K_range)[np.argmax(calinski_scores)]
optimal_k_davies = list(K_range)[np.argmin(davies_scores)]

print(f"\n✓ Optimal K by Silhouette: {optimal_k_silhouette}")
print(f"✓ Optimal K by Calinski-Harabasz: {optimal_k_calinski}")
print(f"✓ Optimal K by Davies-Bouldin: {optimal_k_davies}")

# 6.2 Plot all K selection methods
print("\n6.2 Generating K selection plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Elbow Method
axes[0, 0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0, 0].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[0, 0].set_ylabel('Inertia (WCSS)', fontsize=12)
axes[0, 0].set_title('Elbow Method', fontsize=14, fontweight='bold')
axes[0, 0].axvline(x=5, color='red', linestyle='--', label='Suggested K=5')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Silhouette Score
axes[0, 1].plot(K_range, silhouette_scores_kmeans, 'go-', linewidth=2, markersize=8)
axes[0, 1].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[0, 1].set_ylabel('Silhouette Score', fontsize=12)
axes[0, 1].set_title('Silhouette Score vs K', fontsize=14, fontweight='bold')
axes[0, 1].axvline(x=optimal_k_silhouette, color='red', linestyle='--', label=f'Best K={optimal_k_silhouette}')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Calinski-Harabasz
axes[1, 0].plot(K_range, calinski_scores, 'ro-', linewidth=2, markersize=8)
axes[1, 0].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[1, 0].set_ylabel('Calinski-Harabasz Index', fontsize=12)
axes[1, 0].set_title('Calinski-Harabasz vs K', fontsize=14, fontweight='bold')
axes[1, 0].axvline(x=optimal_k_calinski, color='blue', linestyle='--', label=f'Best K={optimal_k_calinski}')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Davies-Bouldin
axes[1, 1].plot(K_range, davies_scores, 'mo-', linewidth=2, markersize=8)
axes[1, 1].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[1, 1].set_ylabel('Davies-Bouldin Index', fontsize=12)
axes[1, 1].set_title('Davies-Bouldin vs K (Lower = Better)', fontsize=14, fontweight='bold')
axes[1, 1].axvline(x=optimal_k_davies, color='green', linestyle='--', label=f'Best K={optimal_k_davies}')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('04_kmeans_k_selection.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved: 04_kmeans_k_selection.png")

# 6.3 Apply K-Means with K=5
K_OPTIMAL = 5
print(f"\n6.3 Applying K-Means with K={K_OPTIMAL}...")

start_time = time.time()
kmeans_final = KMeans(n_clusters=K_OPTIMAL, random_state=42, n_init=10)
kmeans_labels = kmeans_final.fit_predict(X_scaled)
kmeans_time = time.time() - start_time
kmeans_centroids = kmeans_final.cluster_centers_

print(f"\n✓ K-Means completed in {kmeans_time:.4f} seconds")
print(f"  Iterations: {kmeans_final.n_iter_}")
print(f"  Inertia (WCSS): {kmeans_final.inertia_:.2e}")

print(f"\nK-Means Cluster Distribution:")
for label, count in sorted(Counter(kmeans_labels).items()):
    pct = count / len(kmeans_labels) * 100
    print(f"  Cluster {label}: {count:4} points ({pct:.1f}%)")

# Calculate metrics
kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
kmeans_calinski = calinski_harabasz_score(X_scaled, kmeans_labels)
kmeans_davies = davies_bouldin_score(X_scaled, kmeans_labels)

print(f"\nK-Means Metrics:")
print(f"  Silhouette Score:     {kmeans_silhouette:.4f}")
print(f"  Calinski-Harabasz:    {kmeans_calinski:.2f}")
print(f"  Davies-Bouldin:       {kmeans_davies:.4f}")
print(f"  Runtime:              {kmeans_time:.4f} seconds")

# Add to dataframe
gdf_projected["kmeans_cluster"] = kmeans_labels

# 6.4 Silhouette Analysis per sample
print("\n6.4 Generating Silhouette Analysis plot...")
fig, ax = plt.subplots(figsize=(12, 8))

sample_silhouette_values = silhouette_samples(X_scaled, kmeans_labels)
y_lower = 10

colors = plt.cm.tab10(np.linspace(0, 1, K_OPTIMAL))
for i in range(K_OPTIMAL):
    cluster_silhouette_values = sample_silhouette_values[kmeans_labels == i]
    cluster_silhouette_values.sort()
    
    size_cluster_i = cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_values, 
                     facecolor=colors[i], edgecolor=colors[i], alpha=0.7)
    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10

ax.axvline(x=kmeans_silhouette, color="red", linestyle="--", label=f'Avg: {kmeans_silhouette:.4f}')
ax.set_xlabel("Silhouette Coefficient", fontsize=12)
ax.set_ylabel("Cluster", fontsize=12)
ax.set_title(f"Silhouette Analysis (K-Means, K={K_OPTIMAL})", fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('05_kmeans_silhouette_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved: 05_kmeans_silhouette_analysis.png")

# 6.5 Visualize K-Means clusters
print("\n6.5 Generating K-Means cluster plot...")
fig, ax = plt.subplots(figsize=(14, 10))

for i in range(K_OPTIMAL):
    mask = kmeans_labels == i
    ax.scatter(X[mask, 0], X[mask, 1], c=[colors[i]], s=30, alpha=0.7, 
               edgecolors='black', linewidths=0.3, label=f'Cluster {i} (n={mask.sum()})')

# Plot centroids (transform back from scaled)
centroids_original = scaler.inverse_transform(kmeans_centroids)
ax.scatter(centroids_original[:, 0], centroids_original[:, 1], c='red', marker='X', 
           s=300, edgecolors='black', linewidths=2, label='Centroids', zorder=5)

ax.set_xlabel('X (meters)', fontsize=12)
ax.set_ylabel('Y (meters)', fontsize=12)
ax.set_title(f'K-Means Clustering (K={K_OPTIMAL})\nSilhouette: {kmeans_silhouette:.4f}', fontsize=14, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('06_kmeans_clusters.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved: 06_kmeans_clusters.png")

# =============================================================================
# 7. DBSCAN CLUSTERING
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 7: DBSCAN Clustering")
print("=" * 70)

# 7.1 Finding Optimal eps using k-Distance Graph
min_samples = 10
print(f"\n7.1 Finding Optimal eps (min_samples={min_samples})...")

neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors_fit = neighbors.fit(X)  # Use original coordinates (meters)
distances, _ = neighbors_fit.kneighbors(X)
k_distances = np.sort(distances[:, min_samples - 1])

print(f"\nk-Distance Statistics:")
print(f"  Min: {k_distances[0]:.2f}")
print(f"  Max: {k_distances[-1]:.2f}")
print(f"  Mean: {np.mean(k_distances):.2f}")
print(f"  Median: {np.median(k_distances):.2f}")
print(f"  95th Percentile: {np.percentile(k_distances, 95):.2f}")

# Generate k-Distance plot
print("\n7.2 Generating k-Distance graph...")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(range(len(k_distances)), k_distances, 'b-', linewidth=2)
ax.set_xlabel('Points (sorted)', fontsize=12)
ax.set_ylabel(f'{min_samples}-th Nearest Neighbor Distance', fontsize=12)
ax.set_title(f'k-Distance Graph for DBSCAN (k={min_samples})', fontsize=14, fontweight='bold')

# Mark potential eps values
for eps_val in [600, 800, 1000]:
    ax.axhline(y=eps_val, color='red', linestyle='--', alpha=0.5, label=f'eps={eps_val}')

ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('07_dbscan_kdistance.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved: 07_dbscan_kdistance.png")

# 7.3 Grid Search for DBSCAN parameters
print("\n7.3 Grid Search for DBSCAN parameters...")

eps_values = [400, 600, 800, 1000, 1200]
minPts_values = [5, 10, 15, 20]

results = []
best_sil_dbscan = -1
best_params = {}

print(f"\n{'eps':<8} {'minPts':<8} {'Clusters':<10} {'Noise':<10} {'Noise%':<10} {'Silhouette':<12}")
print("-" * 65)

for eps_val in eps_values:
    for minPts_val in minPts_values:
        dbscan_temp = DBSCAN(eps=eps_val, min_samples=minPts_val)
        labels_temp = dbscan_temp.fit_predict(X)
        
        n_clusters = len(set(labels_temp)) - (1 if -1 in labels_temp else 0)
        n_noise = list(labels_temp).count(-1)
        noise_pct = n_noise / len(labels_temp) * 100
        
        mask_temp = labels_temp != -1
        if len(set(labels_temp[mask_temp])) > 1:
            sil_temp = silhouette_score(X[mask_temp], labels_temp[mask_temp])
            
            if sil_temp > best_sil_dbscan:
                best_sil_dbscan = sil_temp
                best_params = {'eps': eps_val, 'minPts': minPts_val, 
                              'clusters': n_clusters, 'noise': n_noise, 'silhouette': sil_temp}
            
            results.append({
                'eps': eps_val, 'minPts': minPts_val, 'clusters': n_clusters,
                'noise': n_noise, 'noise_pct': noise_pct, 'silhouette': sil_temp
            })
            
            print(f"{eps_val:<8} {minPts_val:<8} {n_clusters:<10} {n_noise:<10} {noise_pct:<10.1f} {sil_temp:<12.4f}")
        else:
            print(f"{eps_val:<8} {minPts_val:<8} {n_clusters:<10} {n_noise:<10} {noise_pct:<10.1f} {'N/A':<12}")

if best_params:
    print(f"\n✓ Best DBSCAN: eps={best_params['eps']}, minPts={best_params['minPts']}")
    print(f"  Clusters: {best_params['clusters']}, Silhouette: {best_params['silhouette']:.4f}")

# 7.4 Apply DBSCAN with best/chosen parameters
EPS_OPTIMAL = 800
MIN_SAMPLES = 15
print(f"\n7.4 Applying DBSCAN (eps={EPS_OPTIMAL}, min_samples={MIN_SAMPLES})...")

start_time = time.time()
dbscan_final = DBSCAN(eps=EPS_OPTIMAL, min_samples=MIN_SAMPLES)
dbscan_labels = dbscan_final.fit_predict(X)
dbscan_time = time.time() - start_time

n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

print(f"\n✓ DBSCAN completed in {dbscan_time:.4f} seconds")
print(f"  Clusters found: {n_clusters_dbscan}")
print(f"  Noise points: {n_noise} ({n_noise/len(X)*100:.1f}%)")

print(f"\nDBSCAN Cluster Distribution:")
for label, count in sorted(Counter(dbscan_labels).items()):
    if label == -1:
        print(f"  Noise:     {count:4} points ({count/len(X)*100:.1f}%)")
    else:
        print(f"  Cluster {label:2}: {count:4} points ({count/len(X)*100:.1f}%)")

# Calculate DBSCAN metrics
mask = dbscan_labels != -1
if len(set(dbscan_labels[mask])) > 1:
    dbscan_silhouette = silhouette_score(X[mask], dbscan_labels[mask])
    dbscan_calinski = calinski_harabasz_score(X[mask], dbscan_labels[mask])
    dbscan_davies = davies_bouldin_score(X[mask], dbscan_labels[mask])
    
    print(f"\nDBSCAN Metrics:")
    print(f"  Silhouette Score:     {dbscan_silhouette:.4f}")
    print(f"  Calinski-Harabasz:    {dbscan_calinski:.2f}")
    print(f"  Davies-Bouldin:       {dbscan_davies:.4f}")
    print(f"  Runtime:              {dbscan_time:.4f} seconds")
else:
    print("\n⚠️ Cannot compute metrics (only 1 cluster or all noise)")
    dbscan_silhouette = -1
    dbscan_calinski = 0
    dbscan_davies = float('inf')

# Add to dataframe
gdf_projected["dbscan_cluster"] = dbscan_labels

# 7.5 Visualize DBSCAN clusters
print("\n7.5 Generating DBSCAN cluster plot...")
fig, ax = plt.subplots(figsize=(14, 10))

unique_labels = sorted(set(dbscan_labels))
colors_db = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

for i, label in enumerate(unique_labels):
    cluster_mask = dbscan_labels == label
    if label == -1:
        ax.scatter(X[cluster_mask, 0], X[cluster_mask, 1], c='black', s=15, 
                   marker='x', alpha=0.5, label=f'Noise (n={cluster_mask.sum()})')
    else:
        ax.scatter(X[cluster_mask, 0], X[cluster_mask, 1], c=[colors_db[i]], s=30, 
                   alpha=0.7, edgecolors='black', linewidths=0.3, 
                   label=f'Cluster {label} (n={cluster_mask.sum()})')

ax.set_xlabel('X (meters)', fontsize=12)
ax.set_ylabel('Y (meters)', fontsize=12)
ax.set_title(f'DBSCAN Clustering (eps={EPS_OPTIMAL}, min_samples={MIN_SAMPLES})\nSilhouette: {dbscan_silhouette:.4f}, Noise: {n_noise} ({n_noise/len(X)*100:.1f}%)', 
             fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('08_dbscan_clusters.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved: 08_dbscan_clusters.png")

# =============================================================================
# 8. SIDE-BY-SIDE COMPARISON
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 8: Side-by-Side Comparison")
print("=" * 70)

print("\nGenerating comparison plot...")
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# Original
axes[0].scatter(X[:, 0], X[:, 1], c='gray', s=15, alpha=0.5)
axes[0].set_xlabel('X (meters)', fontsize=11)
axes[0].set_ylabel('Y (meters)', fontsize=11)
axes[0].set_title(f'Original Data\n({len(X)} Knoxville Amenities)', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# K-Means
for i in range(K_OPTIMAL):
    mask = kmeans_labels == i
    axes[1].scatter(X[mask, 0], X[mask, 1], c=[colors[i]], s=15, alpha=0.7, label=f'C{i}')
axes[1].scatter(centroids_original[:, 0], centroids_original[:, 1], c='red', marker='X', 
                s=200, edgecolors='black', linewidths=2, label='Centers', zorder=5)
axes[1].set_xlabel('X (meters)', fontsize=11)
axes[1].set_ylabel('Y (meters)', fontsize=11)
axes[1].set_title(f'K-Means (K={K_OPTIMAL})\nSilhouette: {kmeans_silhouette:.4f}', fontsize=13, fontweight='bold')
axes[1].legend(loc='upper right', fontsize=8)
axes[1].grid(True, alpha=0.3)

# DBSCAN
for i, label in enumerate(unique_labels):
    cluster_mask = dbscan_labels == label
    if label == -1:
        axes[2].scatter(X[cluster_mask, 0], X[cluster_mask, 1], c='black', s=10, 
                       marker='x', alpha=0.5, label=f'Noise')
    else:
        axes[2].scatter(X[cluster_mask, 0], X[cluster_mask, 1], c=[colors_db[i]], s=15, 
                       alpha=0.7, label=f'C{label}')
axes[2].set_xlabel('X (meters)', fontsize=11)
axes[2].set_ylabel('Y (meters)', fontsize=11)
axes[2].set_title(f'DBSCAN (eps={EPS_OPTIMAL})\nSilhouette: {dbscan_silhouette:.4f}, Noise: {n_noise}', fontsize=13, fontweight='bold')
axes[2].legend(loc='upper right', fontsize=7, ncol=2)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('09_comparison_side_by_side.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved: 09_comparison_side_by_side.png")

# =============================================================================
# 9. METRICS COMPARISON
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 9: Metrics Comparison")
print("=" * 70)

print("\nGenerating metrics comparison plots...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart comparison
metrics_names = ['Silhouette\n(↑ better)', 'Davies-Bouldin\n(↓ better)']
kmeans_vals = [kmeans_silhouette, kmeans_davies]
dbscan_vals = [dbscan_silhouette if dbscan_silhouette > -1 else 0, 
               dbscan_davies if dbscan_davies < float('inf') else 0]

x = np.arange(len(metrics_names))
width = 0.35

bars1 = axes[0].bar(x - width/2, kmeans_vals, width, label='K-Means', color='steelblue', edgecolor='black')
bars2 = axes[0].bar(x + width/2, dbscan_vals, width, label='DBSCAN', color='coral', edgecolor='black')

axes[0].set_ylabel('Score', fontsize=12)
axes[0].set_title('K-Means vs DBSCAN: Metrics Comparison', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics_names)
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Add value labels
for bar in bars1:
    height = bar.get_height()
    axes[0].annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
for bar in bars2:
    height = bar.get_height()
    axes[0].annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

# Runtime comparison
runtimes = [kmeans_time, dbscan_time]
bars3 = axes[1].bar(['K-Means', 'DBSCAN'], runtimes, color=['steelblue', 'coral'], edgecolor='black')
axes[1].set_ylabel('Time (seconds)', fontsize=12)
axes[1].set_title('Runtime Comparison', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

for bar in bars3:
    height = bar.get_height()
    axes[1].annotate(f'{height:.4f}s', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('10_metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved: 10_metrics_comparison.png")

# =============================================================================
# 10. CLUSTER CHARACTERISTICS (AMENITY ANALYSIS)
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 10: Cluster Characteristics")
print("=" * 70)

# Add amenity info to analysis
gdf_projected['amenity'] = gdf['amenity'].values

print("\n10.1 K-Means Cluster Characteristics:")
print("-" * 50)
for i in range(K_OPTIMAL):
    cluster_data = gdf_projected[gdf_projected['kmeans_cluster'] == i]
    top_amenities = cluster_data['amenity'].value_counts().head(5)
    print(f"\nCluster {i} ({len(cluster_data)} points):")
    print(f"  Top amenities:")
    for amenity, count in top_amenities.items():
        print(f"    - {amenity}: {count}")

print("\n10.2 DBSCAN Cluster Characteristics:")
print("-" * 50)
for label in sorted(set(dbscan_labels)):
    if label == -1:
        cluster_data = gdf_projected[gdf_projected['dbscan_cluster'] == -1]
        print(f"\nNoise ({len(cluster_data)} points):")
    else:
        cluster_data = gdf_projected[gdf_projected['dbscan_cluster'] == label]
        print(f"\nCluster {label} ({len(cluster_data)} points):")
    
    top_amenities = cluster_data['amenity'].value_counts().head(3)
    print(f"  Top amenities:")
    for amenity, count in top_amenities.items():
        print(f"    - {amenity}: {count}")

# 10.3 Cluster amenity distribution heatmap
print("\n10.3 Generating cluster-amenity heatmap...")

top_10_amenities = counts.head(10).index.tolist()
cluster_amenity_counts = pd.crosstab(
    gdf_projected['kmeans_cluster'], 
    gdf_projected['amenity']
)[top_10_amenities]

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(cluster_amenity_counts, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
ax.set_xlabel('Amenity Type', fontsize=12)
ax.set_ylabel('K-Means Cluster', fontsize=12)
ax.set_title('Amenity Distribution by K-Means Cluster', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('11_cluster_amenity_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved: 11_cluster_amenity_heatmap.png")

# =============================================================================
# 11. INTERACTIVE MAPS (FOLIUM)
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 11: Interactive Maps")
print("=" * 70)

import folium
from pyproj import Transformer

# Get center of Knoxville
center = [
    gdf_latlon.geometry.y.mean(),
    gdf_latlon.geometry.x.mean()
]

# Create K-Means map
print("\n11.1 Creating K-Means interactive map...")
m_kmeans = folium.Map(location=center, zoom_start=12)

km_colors_hex = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

for idx, row in gdf_latlon.iterrows():
    cluster = gdf_projected.loc[idx, "kmeans_cluster"]
    color = km_colors_hex[cluster % len(km_colors_hex)]
    amenity = gdf_projected.loc[idx, "amenity"]
    
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=5,
        color=color,
        fill=True,
        fillColor=color,
        fillOpacity=0.7,
        popup=f"<b>{amenity}</b><br>Cluster: {cluster}"
    ).add_to(m_kmeans)

# Add centroids
transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
for i, centroid in enumerate(centroids_original):
    lon, lat = transformer.transform(centroid[0], centroid[1])
    folium.Marker(
        location=[lat, lon],
        popup=f"<b>Centroid {i}</b>",
        icon=folium.Icon(color='red', icon='star')
    ).add_to(m_kmeans)

# Add legend
legend_html = '''
<div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; 
     padding: 10px; border: 2px solid gray; border-radius: 5px;">
<b>K-Means Clusters</b><br>
'''
for i in range(K_OPTIMAL):
    legend_html += f'<i style="background:{km_colors_hex[i]}; width:10px; height:10px; display:inline-block;"></i> Cluster {i}<br>'
legend_html += '</div>'
m_kmeans.get_root().html.add_child(folium.Element(legend_html))

m_kmeans.save("12_kmeans_interactive_map.html")
print("✓ Saved: 12_kmeans_interactive_map.html")

# Create DBSCAN map
print("\n11.2 Creating DBSCAN interactive map...")
m_dbscan = folium.Map(location=center, zoom_start=12)

db_colors_hex = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                  '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5']

for idx, row in gdf_latlon.iterrows():
    cluster = gdf_projected.loc[idx, "dbscan_cluster"]
    amenity = gdf_projected.loc[idx, "amenity"]
    
    if cluster == -1:
        color = 'black'
        radius = 3
    else:
        color = db_colors_hex[cluster % len(db_colors_hex)]
        radius = 5
    
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=radius,
        color=color,
        fill=True,
        fillColor=color,
        fillOpacity=0.7,
        popup=f"<b>{amenity}</b><br>Cluster: {'Noise' if cluster == -1 else cluster}"
    ).add_to(m_dbscan)

m_dbscan.save("13_dbscan_interactive_map.html")
print("✓ Saved: 13_dbscan_interactive_map.html")

# =============================================================================
# 12. SAVE RESULTS
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 12: Save Results")
print("=" * 70)

# Create results dataframe
results_df = pd.DataFrame({
    'amenity': gdf_projected['amenity'],
    'x_meters': X[:, 0],
    'y_meters': X[:, 1],
    'latitude': gdf_latlon.geometry.y.values,
    'longitude': gdf_latlon.geometry.x.values,
    'kmeans_cluster': kmeans_labels,
    'dbscan_cluster': dbscan_labels
})

results_df.to_csv('knoxville_clustering_results.csv', index=False)
print("✓ Saved: knoxville_clustering_results.csv")

# =============================================================================
# 13. FINAL COMPARISON TABLE
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 13: Final Comparison")
print("=" * 70)

# Determine winner
winner = "DBSCAN" if dbscan_silhouette > kmeans_silhouette else "K-Means"

print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║          K-MEANS vs DBSCAN: KNOXVILLE AMENITIES CLUSTERING               ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  DATASET: {len(X)} amenities in Knoxville, Tennessee (OpenStreetMap)       ║
║  Features: Latitude, Longitude (2D spatial data)                         ║
║  Top amenities: restaurants, churches, parking, fast food, cafes         ║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  METRIC                    K-MEANS (K={K_OPTIMAL})      DBSCAN (ε={EPS_OPTIMAL})          ║
║  ──────────────────────────────────────────────────────────────────      ║
║  Silhouette Score          {kmeans_silhouette:.4f}              {dbscan_silhouette:.4f}              ║
║  Calinski-Harabasz         {kmeans_calinski:<10.2f}         {dbscan_calinski:<10.2f}         ║
║  Davies-Bouldin            {kmeans_davies:.4f}              {dbscan_davies:.4f}              ║
║  Clusters Found            {K_OPTIMAL:<5}               {n_clusters_dbscan:<5}               ║
║  Noise Points              0                   {n_noise} ({n_noise/len(X)*100:.1f}%)            ║
║  Points Clustered          {len(X):<5}               {len(X) - n_noise:<5}               ║
║  Runtime (seconds)         {kmeans_time:.4f}             {dbscan_time:.4f}             ║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║  TIME COMPLEXITY:                                                        ║
║  • K-Means: O(n × K × d × i) = O({len(X)} × {K_OPTIMAL} × 2 × {kmeans_final.n_iter_})              ║
║  • DBSCAN:  O(n²) worst case, O(n log n) with spatial index              ║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║  🏆 WINNER: {winner:<10} (by Silhouette Score)                            ║
║                                                                          ║
║  WHY {winner} WON:                                                        ║
║  • {'DBSCAN detected ' + str(n_noise) + ' outliers (isolated amenities)' if winner == 'DBSCAN' else 'K-Means clustered all points evenly'}          ║
║  • {'Better handles irregular city density patterns' if winner == 'DBSCAN' else 'Good for creating balanced service zones'}                    ║
║  • {'Finds natural neighborhoods without specifying K' if winner == 'DBSCAN' else 'Simple, fast, and interpretable'}                      ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

# =============================================================================
# 14. METHODS SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 14: Methods Summary")
print("=" * 70)

print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                    METHODS TO CHOOSE PARAMETERS                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  METHOD              ALGORITHM    FINDS        HOW TO USE                ║
║  ──────────────────────────────────────────────────────────────────      ║
║  Elbow (WCSS)        K-Means      K            Plot WCSS, find bend      ║
║  k-Distance Graph    DBSCAN       ε            Plot k-dist, find bend    ║
║  Rule of Thumb       DBSCAN       minPts       minPts = 2 × dimensions   ║
║  Silhouette          Both         Best params  Try values, pick highest  ║
║  Calinski-Harabasz   Both         Best params  Try values, pick highest  ║
║  Davies-Bouldin      Both         Best params  Try values, pick lowest   ║
║  Grid Search         DBSCAN       ε & minPts   Try all combinations      ║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║                    EVALUATION METRICS                                    ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  METHOD              FORMULA                    GOAL        RANGE        ║
║  ──────────────────────────────────────────────────────────────────      ║
║  Silhouette          s = (b-a)/max(a,b)         Maximize    -1 to 1      ║
║  Calinski-Harabasz   CH = [Bk/(K-1)]/[Wk/(n-K)] Maximize    0 to ∞       ║
║  Davies-Bouldin      DB = (1/K)Σmax(Rij)        Minimize    0 to ∞       ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

# =============================================================================
# 15. FILES SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 15: Files Created")
print("=" * 70)

print("""
VISUALIZATION FILES:
  ✓ 01_spatial_distribution.png     - Original data distribution
  ✓ 02_amenity_distribution.png     - Top 15 amenity types bar chart
  ✓ 03_density_heatmap.png          - Density heatmap
  ✓ 04_kmeans_k_selection.png       - All K selection methods (4 plots)
  ✓ 05_kmeans_silhouette_analysis.png - Per-sample silhouette analysis
  ✓ 06_kmeans_clusters.png          - K-Means clustering result
  ✓ 07_dbscan_kdistance.png         - k-Distance graph for DBSCAN
  ✓ 08_dbscan_clusters.png          - DBSCAN clustering result
  ✓ 09_comparison_side_by_side.png  - Original vs K-Means vs DBSCAN
  ✓ 10_metrics_comparison.png       - Metrics & runtime comparison
  ✓ 11_cluster_amenity_heatmap.png  - Amenity distribution by cluster

INTERACTIVE MAPS (open in browser):
  ✓ 12_kmeans_interactive_map.html  - K-Means clusters on map
  ✓ 13_dbscan_interactive_map.html  - DBSCAN clusters on map

DATA FILES:
  ✓ knoxville_clustering_results.csv - All results with cluster labels
""")

print("\n" + "=" * 70)
print("SCRIPT COMPLETED SUCCESSFULLY! ✓")
print("=" * 70)
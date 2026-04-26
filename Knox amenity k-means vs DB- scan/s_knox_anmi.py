"""
================================================================================
K-Means vs DBSCAN: Knoxville Amenities Clustering
WITH ANIMATIONS & ENHANCED MAPS (FULLY FIXED VERSION)
================================================================================
"""

# =============================================================================
# 1. IMPORT LIBRARIES
# =============================================================================
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import warnings
import time
import imageio
import os

warnings.filterwarnings('ignore')

print("=" * 70)
print("K-Means vs DBSCAN: Knoxville Amenities Clustering")
print("WITH ANIMATIONS & ENHANCED MAPS")
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

print("Fetching amenities from OpenStreetMap API...")
start_time = time.time()
area = ox.geocode_to_gdf(place)
tags = {"amenity": True}
gdf = ox.features_from_polygon(area.geometry.iloc[0], tags)
fetch_time = time.time() - start_time

gdf = gdf[gdf.geometry.type == "Point"].copy()
gdf = gdf[~gdf["amenity"].isin(["street_lamp", "letter_box"])]

print(f"✓ Data fetched in {fetch_time:.2f} seconds")
print(f"✓ Total amenities loaded: {len(gdf)}")

# Prepare data
gdf_projected = gdf.to_crs(epsg=3857)
X = np.vstack([gdf_projected.geometry.x, gdf_projected.geometry.y]).T

gdf_latlon = gdf.to_crs(epsg=4326)
gdf_projected['amenity'] = gdf['amenity'].values

print(f"✓ Coordinates extracted: {X.shape[0]} points")

counts = gdf["amenity"].value_counts()

# =============================================================================
# 3. K-MEANS ANIMATION (GIF) - FIXED
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 3: Creating K-Means Animation (GIF)")
print("=" * 70)

def create_kmeans_animation(X, n_clusters=5, max_iter=15):
    """Create step-by-step K-Means animation with fixed frame sizes"""
    
    print(f"\nCreating K-Means animation with K={n_clusters}...")
    
    # Normalize for visualization
    X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    
    # Initialize centroids randomly
    np.random.seed(42)
    centroid_indices = np.random.choice(len(X_norm), n_clusters, replace=False)
    centroids = X_norm[centroid_indices].copy()
    
    # Store history
    centroid_history = [centroids.copy()]
    label_history = []
    
    # Run K-Means manually
    for iteration in range(max_iter):
        distances = np.sqrt(((X_norm[:, np.newaxis] - centroids) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=1)
        label_history.append(labels.copy())
        
        new_centroids = np.array([X_norm[labels == k].mean(axis=0) if (labels == k).sum() > 0 
                                   else centroids[k] for k in range(n_clusters)])
        
        if np.allclose(centroids, new_centroids):
            print(f"  Converged at iteration {iteration + 1}")
            break
            
        centroids = new_centroids.copy()
        centroid_history.append(centroids.copy())
    
    # Create frames with FIXED SIZE
    frames = []
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    # Fixed figure size and DPI for consistent frame sizes
    fig_width, fig_height = 8, 6
    dpi = 100
    
    for i, (centroids, labels) in enumerate(zip(centroid_history, label_history)):
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        
        # Plot points
        for k in range(n_clusters):
            mask = labels == k
            ax.scatter(X_norm[mask, 0], X_norm[mask, 1], c=[colors[k]], s=15, alpha=0.6, 
                      edgecolors='black', linewidths=0.2)
        
        # Plot centroids
        ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=300, 
                  edgecolors='black', linewidths=2, zorder=5, label='Centroids')
        
        # Draw circles around centroids
        for k, centroid in enumerate(centroids):
            circle = Circle(centroid, 0.12, fill=False, color=colors[k], linewidth=2, linestyle='--')
            ax.add_patch(circle)
        
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('X (normalized)', fontsize=12)
        ax.set_ylabel('Y (normalized)', fontsize=12)
        ax.set_title(f'K-Means Clustering - Iteration {i + 1}\n'
                    f'({len(X)} Knoxville Amenities, K={n_clusters})', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.set_aspect('equal')
        
        # Save to array directly with fixed size
        fig.canvas.draw()
        
        # Convert to numpy array with consistent size
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = img[:, :, :3]  # Remove alpha channel
        
        frames.append(img.copy())
        plt.close(fig)
    
    # Save as GIF
    imageio.mimsave('animation_kmeans.gif', frames, duration=1.0, loop=0)
    print("✓ Saved: animation_kmeans.gif")
    
    return centroid_history, label_history

centroid_history, label_history = create_kmeans_animation(X, n_clusters=5, max_iter=15)

# =============================================================================
# 4. K-MEANS STEP-BY-STEP IMAGES
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 4: K-Means Step-by-Step Images")
print("=" * 70)

def create_kmeans_steps_image(X, n_clusters=5, max_iter=6):
    """Create step-by-step K-Means visualization"""
    
    print("\nCreating K-Means step-by-step images...")
    
    X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    
    np.random.seed(42)
    centroid_indices = np.random.choice(len(X_norm), n_clusters, replace=False)
    centroids = X_norm[centroid_indices].copy()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    for iteration in range(min(max_iter, 6)):
        ax = axes[iteration]
        
        distances = np.sqrt(((X_norm[:, np.newaxis] - centroids) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=1)
        
        for k in range(n_clusters):
            mask = labels == k
            ax.scatter(X_norm[mask, 0], X_norm[mask, 1], c=[colors[k]], s=15, alpha=0.6,
                      edgecolors='black', linewidths=0.1)
        
        ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200,
                  edgecolors='black', linewidths=2, zorder=5)
        
        ax.set_title(f'Iteration {iteration + 1}', fontsize=12, fontweight='bold')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        
        centroids = np.array([X_norm[labels == k].mean(axis=0) if (labels == k).sum() > 0 
                              else centroids[k] for k in range(n_clusters)])
    
    plt.suptitle('K-Means Algorithm: Step-by-Step\n(Red X = Centroids moving to cluster centers)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('kmeans_steps.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: kmeans_steps.png")

create_kmeans_steps_image(X, n_clusters=5)

# =============================================================================
# 5. DBSCAN ANIMATION (GIF) - FIXED
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 5: Creating DBSCAN Animation (GIF)")
print("=" * 70)

def create_dbscan_animation(X, eps=0.08, min_samples=5):
    """Create step-by-step DBSCAN animation with fixed frame sizes"""
    
    print(f"\nCreating DBSCAN animation (eps={eps}, min_samples={min_samples})...")
    
    X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    
    n_points = len(X_norm)
    labels = np.full(n_points, -2)  # -2 = unvisited, -1 = noise
    cluster_id = 0
    
    frames = []
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    
    # Fixed figure size
    fig_width, fig_height = 8, 6
    dpi = 100
    
    def get_neighbors(point_idx):
        distances = np.sqrt(((X_norm - X_norm[point_idx]) ** 2).sum(axis=1))
        return np.where(distances <= eps)[0]
    
    def save_frame(title, current_point=None, neighbors=None):
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        
        # Plot unvisited points
        unvisited_mask = labels == -2
        ax.scatter(X_norm[unvisited_mask, 0], X_norm[unvisited_mask, 1], 
                  c='lightgray', s=15, alpha=0.5, label='Unvisited')
        
        # Plot noise points
        noise_mask = labels == -1
        if noise_mask.sum() > 0:
            ax.scatter(X_norm[noise_mask, 0], X_norm[noise_mask, 1],
                      c='black', s=20, marker='x', alpha=0.7, label='Noise')
        
        # Plot clustered points
        for c in range(cluster_id + 1):
            cluster_mask = labels == c
            if cluster_mask.sum() > 0:
                ax.scatter(X_norm[cluster_mask, 0], X_norm[cluster_mask, 1],
                          c=[colors[c % 20]], s=20, alpha=0.7, edgecolors='black', 
                          linewidths=0.2, label=f'Cluster {c}')
        
        # Highlight current point
        if current_point is not None:
            ax.scatter(X_norm[current_point, 0], X_norm[current_point, 1],
                      c='red', s=150, marker='*', edgecolors='black', linewidths=2,
                      zorder=10, label='Current Point')
            
            circle = Circle(X_norm[current_point], eps, fill=False, color='red', 
                           linewidth=2, linestyle='--')
            ax.add_patch(circle)
        
        # Highlight neighbors
        if neighbors is not None and len(neighbors) > 0:
            ax.scatter(X_norm[neighbors, 0], X_norm[neighbors, 1],
                      c='yellow', s=40, marker='o', edgecolors='red', linewidths=1,
                      zorder=9, label=f'Neighbors ({len(neighbors)})')
        
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('X (normalized)', fontsize=12)
        ax.set_ylabel('Y (normalized)', fontsize=12)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=7)
        ax.set_aspect('equal')
        
        # Save to array directly
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = img[:, :, :3]
        
        frames.append(img.copy())
        plt.close(fig)
    
    # Initial frame
    save_frame('DBSCAN: Initial State - All Points Unvisited')
    
    # Sample points for animation
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_norm), min(25, len(X_norm)), replace=False)
    
    for idx in sample_indices:
        if labels[idx] != -2:
            continue
            
        neighbors = get_neighbors(idx)
        
        if len(neighbors) >= min_samples:
            labels[idx] = cluster_id
            save_frame(f'DBSCAN: Core Point Found (Cluster {cluster_id})\n'
                      f'Neighbors: {len(neighbors)} >= minPts ({min_samples})',
                      current_point=idx, neighbors=neighbors)
            
            for neighbor in neighbors:
                if labels[neighbor] == -2:
                    labels[neighbor] = cluster_id
            
            cluster_id += 1
        else:
            labels[idx] = -1
            save_frame(f'DBSCAN: Noise Point Found\n'
                      f'Neighbors: {len(neighbors)} < minPts ({min_samples})',
                      current_point=idx, neighbors=neighbors)
    
    # Final frame
    save_frame(f'DBSCAN: Complete\n{cluster_id} Clusters, Noise: {(labels == -1).sum()}')
    
    # Save as GIF
    imageio.mimsave('animation_dbscan.gif', frames, duration=0.8, loop=0)
    print("✓ Saved: animation_dbscan.gif")

create_dbscan_animation(X, eps=0.08, min_samples=5)

# =============================================================================
# 6. DBSCAN CONCEPT IMAGE
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 6: DBSCAN Concept Illustration")
print("=" * 70)

def create_dbscan_concept_image():
    """Create DBSCAN concept visualization"""
    
    print("\nCreating DBSCAN concept visualization...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    np.random.seed(42)
    
    # Core point
    ax = axes[0]
    center = np.array([0.5, 0.5])
    neighbors = center + np.random.randn(8, 2) * 0.1
    ax.scatter(neighbors[:, 0], neighbors[:, 1], c='blue', s=100, alpha=0.7, 
              edgecolors='black', linewidths=1, label='Neighbors (8)')
    ax.scatter(center[0], center[1], c='red', s=200, marker='*', 
              edgecolors='black', linewidths=2, label='Core Point', zorder=5)
    circle = Circle(center, 0.15, fill=False, color='red', linewidth=2, linestyle='--')
    ax.add_patch(circle)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Core Point\n(>= minPts neighbors)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.set_aspect('equal')
    ax.text(0.5, 0.1, '8 neighbors >= minPts(5) ✓', ha='center', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='lightgreen'))
    
    # Border point
    ax = axes[1]
    core = np.array([0.4, 0.5])
    border = np.array([0.6, 0.5])
    other_neighbors = core + np.random.randn(6, 2) * 0.08
    ax.scatter(other_neighbors[:, 0], other_neighbors[:, 1], c='blue', s=100, alpha=0.7,
              edgecolors='black', linewidths=1)
    ax.scatter(core[0], core[1], c='red', s=200, marker='*',
              edgecolors='black', linewidths=2, label='Core Point', zorder=5)
    ax.scatter(border[0], border[1], c='orange', s=200, marker='s',
              edgecolors='black', linewidths=2, label='Border Point', zorder=5)
    circle1 = Circle(core, 0.15, fill=False, color='red', linewidth=2, linestyle='--')
    circle2 = Circle(border, 0.15, fill=False, color='orange', linewidth=2, linestyle='--')
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Border Point\n(< minPts, but near core)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.set_aspect('equal')
    ax.text(0.5, 0.1, '2 neighbors < minPts, in core ε', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
    
    # Noise point
    ax = axes[2]
    cluster = np.array([[0.3, 0.5], [0.35, 0.55], [0.4, 0.45], [0.32, 0.48], [0.38, 0.52]])
    noise = np.array([0.8, 0.8])
    ax.scatter(cluster[:, 0], cluster[:, 1], c='blue', s=100, alpha=0.7,
              edgecolors='black', linewidths=1, label='Cluster')
    ax.scatter(noise[0], noise[1], c='black', s=200, marker='X',
              linewidths=3, label='Noise Point', zorder=5)
    circle = Circle(noise, 0.15, fill=False, color='black', linewidth=2, linestyle='--')
    ax.add_patch(circle)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Noise Point\n(< minPts, not near core)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.set_aspect('equal')
    ax.text(0.5, 0.1, '0 neighbors, isolated ✗', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightcoral'))
    
    plt.suptitle('DBSCAN: Point Classification (minPts=5)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('dbscan_concept.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: dbscan_concept.png")

create_dbscan_concept_image()

# =============================================================================
# 7. RUN ACTUAL CLUSTERING
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 7: Running Clustering Algorithms")
print("=" * 70)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means
K_OPTIMAL = 5
print(f"\nRunning K-Means (K={K_OPTIMAL})...")
start_time = time.time()
kmeans = KMeans(n_clusters=K_OPTIMAL, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
kmeans_time = time.time() - start_time
kmeans_centroids = scaler.inverse_transform(kmeans.cluster_centers_)

kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
kmeans_calinski = calinski_harabasz_score(X_scaled, kmeans_labels)
kmeans_davies = davies_bouldin_score(X_scaled, kmeans_labels)

print(f"✓ K-Means: Silhouette={kmeans_silhouette:.4f}, Time={kmeans_time:.4f}s")

# DBSCAN
EPS_OPTIMAL = 800
MIN_SAMPLES = 10
print(f"\nRunning DBSCAN (eps={EPS_OPTIMAL}, min_samples={MIN_SAMPLES})...")
start_time = time.time()
dbscan = DBSCAN(eps=EPS_OPTIMAL, min_samples=MIN_SAMPLES)
dbscan_labels = dbscan.fit_predict(X)
dbscan_time = time.time() - start_time

n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

mask = dbscan_labels != -1
if len(set(dbscan_labels[mask])) > 1:
    dbscan_silhouette = silhouette_score(X[mask], dbscan_labels[mask])
    dbscan_calinski = calinski_harabasz_score(X[mask], dbscan_labels[mask])
    dbscan_davies = davies_bouldin_score(X[mask], dbscan_labels[mask])
else:
    dbscan_silhouette = -1
    dbscan_calinski = 0
    dbscan_davies = float('inf')

print(f"✓ DBSCAN: Silhouette={dbscan_silhouette:.4f}, Clusters={n_clusters_dbscan}, Noise={n_noise}")

gdf_projected["kmeans_cluster"] = kmeans_labels
gdf_projected["dbscan_cluster"] = dbscan_labels

# =============================================================================
# 8. COMPARISON ANIMATION (GIF)
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 8: Creating Comparison Animation")
print("=" * 70)

def create_comparison_animation():
    """Create animation comparing original, K-Means, and DBSCAN"""
    
    print("\nCreating comparison animation...")
    
    X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    
    frames = []
    colors_km = plt.cm.tab10(np.linspace(0, 1, K_OPTIMAL))
    colors_db = plt.cm.tab20(np.linspace(0, 1, 20))
    
    fig_width, fig_height = 8, 6
    dpi = 100
    
    # Frame 1: Original
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax.scatter(X_norm[:, 0], X_norm[:, 1], c='gray', s=15, alpha=0.6)
    ax.set_title(f'Original Data\n({len(X)} Knoxville Amenities)', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    frames.extend([img.copy()] * 2)
    plt.close()
    
    # Frame 2: K-Means
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    for k in range(K_OPTIMAL):
        mask = kmeans_labels == k
        ax.scatter(X_norm[mask, 0], X_norm[mask, 1], c=[colors_km[k]], s=15, alpha=0.7)
    centroids_norm = (kmeans_centroids - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    ax.scatter(centroids_norm[:, 0], centroids_norm[:, 1], c='red', marker='X', s=200,
              edgecolors='black', linewidths=2, zorder=5)
    ax.set_title(f'K-Means (K={K_OPTIMAL})\nSilhouette: {kmeans_silhouette:.4f}', 
                fontsize=14, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    frames.extend([img.copy()] * 2)
    plt.close()
    
    # Frame 3: DBSCAN
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    for label in sorted(set(dbscan_labels)):
        mask = dbscan_labels == label
        if label == -1:
            ax.scatter(X_norm[mask, 0], X_norm[mask, 1], c='black', s=8, marker='x', alpha=0.5)
        else:
            ax.scatter(X_norm[mask, 0], X_norm[mask, 1], c=[colors_db[label % 20]], s=15, alpha=0.7)
    ax.set_title(f'DBSCAN (eps={EPS_OPTIMAL})\nSilhouette: {dbscan_silhouette:.4f}, Noise: {n_noise}', 
                fontsize=14, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    frames.extend([img.copy()] * 2)
    plt.close()
    
    imageio.mimsave('animation_comparison.gif', frames, duration=1.5, loop=0)
    print("✓ Saved: animation_comparison.gif")

create_comparison_animation()

# =============================================================================
# 9. STATIC COMPARISON IMAGE
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 9: Creating Static Comparison Image")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
colors_km = plt.cm.tab10(np.linspace(0, 1, K_OPTIMAL))
colors_db = plt.cm.tab20(np.linspace(0, 1, 20))

# Original
axes[0].scatter(X[:, 0], X[:, 1], c='gray', s=10, alpha=0.5)
axes[0].set_title(f'Original Data\n({len(X)} Knoxville Amenities)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('X (meters)')
axes[0].set_ylabel('Y (meters)')

# K-Means
for k in range(K_OPTIMAL):
    mask = kmeans_labels == k
    axes[1].scatter(X[mask, 0], X[mask, 1], c=[colors_km[k]], s=10, alpha=0.7, label=f'C{k}')
axes[1].scatter(kmeans_centroids[:, 0], kmeans_centroids[:, 1], c='red', marker='X', 
                s=200, edgecolors='black', linewidths=2, zorder=5)
axes[1].set_title(f'K-Means (K={K_OPTIMAL})\nSilhouette: {kmeans_silhouette:.4f}', fontsize=13, fontweight='bold')
axes[1].set_xlabel('X (meters)')
axes[1].legend(loc='upper right', fontsize=8)

# DBSCAN
for label in sorted(set(dbscan_labels)):
    mask = dbscan_labels == label
    if label == -1:
        axes[2].scatter(X[mask, 0], X[mask, 1], c='black', s=5, marker='x', alpha=0.5, label='Noise')
    elif label < 5:
        axes[2].scatter(X[mask, 0], X[mask, 1], c=[colors_db[label % 20]], s=10, alpha=0.7, label=f'C{label}')
    else:
        axes[2].scatter(X[mask, 0], X[mask, 1], c=[colors_db[label % 20]], s=10, alpha=0.7)
axes[2].set_title(f'DBSCAN (eps={EPS_OPTIMAL})\nSilhouette: {dbscan_silhouette:.4f}, Noise: {n_noise}', 
                  fontsize=13, fontweight='bold')
axes[2].set_xlabel('X (meters)')
axes[2].legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig('comparison_final.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: comparison_final.png")

# =============================================================================
# 10. INTERACTIVE MAPS
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 10: Creating Interactive Maps")
print("=" * 70)

import folium
from folium.plugins import HeatMap, MiniMap, DualMap
from pyproj import Transformer

center = [gdf_latlon.geometry.y.mean(), gdf_latlon.geometry.x.mean()]

# 10.1 All Data Map
print("\n10.1 Creating All Data Map...")
m_all = folium.Map(location=center, zoom_start=12, tiles='CartoDB positron')
MiniMap().add_to(m_all)

amenity_colors = {
    'restaurant': 'red', 'place_of_worship': 'blue', 'fast_food': 'orange',
    'cafe': 'brown', 'parking': 'gray', 'school': 'green', 'fuel': 'black',
    'bar': 'purple', 'bank': 'darkblue', 'pharmacy': 'pink'
}

for idx, row in gdf_latlon.iterrows():
    amenity = gdf_projected.loc[idx, "amenity"]
    color = amenity_colors.get(amenity, 'lightgray')
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x], radius=4,
        color=color, fill=True, fillColor=color, fillOpacity=0.7,
        popup=f"<b>{amenity}</b>"
    ).add_to(m_all)

m_all.save("map_all_amenities.html")
print("✓ Saved: map_all_amenities.html")

# 10.2 Heatmap
print("\n10.2 Creating Heatmap...")
m_heat = folium.Map(location=center, zoom_start=12, tiles='CartoDB dark_matter')
heat_data = [[row.geometry.y, row.geometry.x] for idx, row in gdf_latlon.iterrows()]
HeatMap(heat_data, radius=15, blur=10, max_zoom=13).add_to(m_heat)
m_heat.save("map_heatmap.html")
print("✓ Saved: map_heatmap.html")

# 10.3 K-Means Map
print("\n10.3 Creating K-Means Map...")
m_km = folium.Map(location=center, zoom_start=12)
km_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

for idx, row in gdf_latlon.iterrows():
    cluster = gdf_projected.loc[idx, "kmeans_cluster"]
    amenity = gdf_projected.loc[idx, "amenity"]
    color = km_colors[cluster % len(km_colors)]
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x], radius=5,
        color=color, fill=True, fillColor=color, fillOpacity=0.7,
        popup=f"<b>{amenity}</b><br>Cluster: {cluster}"
    ).add_to(m_km)

transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
for i, centroid in enumerate(kmeans_centroids):
    lon, lat = transformer.transform(centroid[0], centroid[1])
    folium.Marker(
        location=[lat, lon], popup=f"<b>Centroid {i}</b>",
        icon=folium.Icon(color='red', icon='star')
    ).add_to(m_km)

m_km.save("map_kmeans.html")
print("✓ Saved: map_kmeans.html")

# 10.4 DBSCAN Map
print("\n10.4 Creating DBSCAN Map...")
m_db = folium.Map(location=center, zoom_start=12)
db_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

for idx, row in gdf_latlon.iterrows():
    cluster = gdf_projected.loc[idx, "dbscan_cluster"]
    amenity = gdf_projected.loc[idx, "amenity"]
    if cluster == -1:
        color = 'black'
        radius = 3
    else:
        color = db_colors[cluster % len(db_colors)]
        radius = 5
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x], radius=radius,
        color=color, fill=True, fillColor=color, fillOpacity=0.7,
        popup=f"<b>{amenity}</b><br>Cluster: {'Noise' if cluster == -1 else cluster}"
    ).add_to(m_db)

m_db.save("map_dbscan.html")
print("✓ Saved: map_dbscan.html")

# 10.5 Side-by-Side Map
print("\n10.5 Creating Side-by-Side Map...")
m_dual = DualMap(location=center, zoom_start=12)

for idx, row in gdf_latlon.iterrows():
    cluster = gdf_projected.loc[idx, "kmeans_cluster"]
    color = km_colors[cluster % len(km_colors)]
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x], radius=4,
        color=color, fill=True, fillColor=color, fillOpacity=0.7
    ).add_to(m_dual.m1)

for idx, row in gdf_latlon.iterrows():
    cluster = gdf_projected.loc[idx, "dbscan_cluster"]
    color = 'black' if cluster == -1 else db_colors[cluster % len(db_colors)]
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x], radius=4,
        color=color, fill=True, fillColor=color, fillOpacity=0.7
    ).add_to(m_dual.m2)

m_dual.save("map_comparison_dual.html")
print("✓ Saved: map_comparison_dual.html")

# =============================================================================
# 11. FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 11: Final Summary")
print("=" * 70)

winner = "DBSCAN" if dbscan_silhouette > kmeans_silhouette else "K-Means"

print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║          K-MEANS vs DBSCAN: KNOXVILLE AMENITIES                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Dataset: {len(X)} amenities                                               ║
║                                                                          ║
║  METRIC                    K-MEANS          DBSCAN                       ║
║  Silhouette Score          {kmeans_silhouette:.4f}           {dbscan_silhouette:.4f}                       ║
║  Clusters                  {K_OPTIMAL}                {n_clusters_dbscan}                          ║
║  Noise                     0                {n_noise}                          ║
║                                                                          ║
║  🏆 WINNER: {winner}                                                      ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print("""
ANIMATIONS (GIF):
  ✓ animation_kmeans.gif       - K-Means step-by-step
  ✓ animation_dbscan.gif       - DBSCAN step-by-step
  ✓ animation_comparison.gif   - Original → K-Means → DBSCAN

IMAGES:
  ✓ kmeans_steps.png           - K-Means 6 iterations
  ✓ dbscan_concept.png         - Core/Border/Noise illustration
  ✓ comparison_final.png       - Side-by-side comparison

INTERACTIVE MAPS:
  ✓ map_all_amenities.html     - All amenities by type
  ✓ map_heatmap.html           - Density heatmap
  ✓ map_kmeans.html            - K-Means clusters
  ✓ map_dbscan.html            - DBSCAN clusters
  ✓ map_comparison_dual.html   - Side-by-side comparison
""")

print("\n" + "=" * 70)
print("SCRIPT COMPLETED SUCCESSFULLY! ✓")
print("=" * 70)
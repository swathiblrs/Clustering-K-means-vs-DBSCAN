"""
================================================================================
K-Means vs DBSCAN: Knoxville Amenities Clustering
FINAL VERSION - SLOW ANIMATIONS + MAP VISUALIZATIONS
================================================================================
For CS 581 Presentation
- Slow GIFs for audience explanation
- Clustering on actual Knoxville maps
- All interactive maps
================================================================================
"""

# =============================================================================
# 1. IMPORT LIBRARIES
# =============================================================================
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
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
print("SLOW ANIMATIONS + MAP VISUALIZATIONS")
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

# Get lat/lon coordinates
lats = gdf_latlon.geometry.y.values
lons = gdf_latlon.geometry.x.values

print(f"✓ Coordinates extracted: {X.shape[0]} points")

counts = gdf["amenity"].value_counts()

# =============================================================================
# 3. RUN CLUSTERING FIRST
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 3: Running Clustering Algorithms")
print("=" * 70)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means
K_OPTIMAL = 5
print(f"\nRunning K-Means (K={K_OPTIMAL})...")
kmeans = KMeans(n_clusters=K_OPTIMAL, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
kmeans_centroids = scaler.inverse_transform(kmeans.cluster_centers_)
kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
print(f"✓ K-Means: Silhouette={kmeans_silhouette:.4f}")

# DBSCAN
EPS_OPTIMAL = 800
MIN_SAMPLES = 10
print(f"\nRunning DBSCAN (eps={EPS_OPTIMAL}, min_samples={MIN_SAMPLES})...")
dbscan = DBSCAN(eps=EPS_OPTIMAL, min_samples=MIN_SAMPLES)
dbscan_labels = dbscan.fit_predict(X)
n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

mask = dbscan_labels != -1
if len(set(dbscan_labels[mask])) > 1:
    dbscan_silhouette = silhouette_score(X[mask], dbscan_labels[mask])
else:
    dbscan_silhouette = -1
print(f"✓ DBSCAN: Silhouette={dbscan_silhouette:.4f}, Clusters={n_clusters_dbscan}, Noise={n_noise}")

gdf_projected["kmeans_cluster"] = kmeans_labels
gdf_projected["dbscan_cluster"] = dbscan_labels
gdf_latlon["kmeans_cluster"] = kmeans_labels
gdf_latlon["dbscan_cluster"] = dbscan_labels

# =============================================================================
# 4. K-MEANS SLOW ANIMATION (3 seconds per frame)
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 4: Creating SLOW K-Means Animation")
print("=" * 70)

def create_kmeans_slow_animation(X, lats, lons, n_clusters=5, max_iter=15):
    """Create SLOW K-Means animation for presentation"""
    
    print(f"\nCreating SLOW K-Means animation (3 sec/frame)...")
    
    # Normalize coordinates
    X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    
    # Use lat/lon for visualization (more intuitive)
    lat_norm = (lats - lats.min()) / (lats.max() - lats.min())
    lon_norm = (lons - lons.min()) / (lons.max() - lons.min())
    
    # Initialize centroids
    np.random.seed(42)
    centroid_indices = np.random.choice(len(X_norm), n_clusters, replace=False)
    centroids_x = lon_norm[centroid_indices].copy()
    centroids_y = lat_norm[centroid_indices].copy()
    
    frames = []
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    
    fig_width, fig_height = 10, 8
    dpi = 100
    
    # Frame 0: Initial random centroids
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax.scatter(lon_norm, lat_norm, c='gray', s=15, alpha=0.5, label='Amenities')
    ax.scatter(centroids_x, centroids_y, c='red', marker='X', s=400, 
              edgecolors='black', linewidths=3, zorder=5, label='Initial Centroids')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Longitude (normalized)', fontsize=12)
    ax.set_ylabel('Latitude (normalized)', fontsize=12)
    ax.set_title('K-Means: Step 1 - Random Initial Centroids\n'
                f'({len(X)} Knoxville Amenities)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.text(0.5, -0.12, 'Centroids placed randomly in the data space', 
            ha='center', transform=ax.transAxes, fontsize=11, style='italic')
    
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    frames.append(img.copy())
    plt.close()
    
    # Iterations
    for iteration in range(min(max_iter, 8)):
        # Assign points to nearest centroid
        distances = np.sqrt((lon_norm[:, np.newaxis] - centroids_x) ** 2 + 
                           (lat_norm[:, np.newaxis] - centroids_y) ** 2)
        labels = np.argmin(distances, axis=1)
        
        # Frame: Show assignment
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        
        for k in range(n_clusters):
            cluster_mask = labels == k
            ax.scatter(lon_norm[cluster_mask], lat_norm[cluster_mask], 
                      c=colors[k], s=15, alpha=0.7, label=f'Cluster {k}')
        
        ax.scatter(centroids_x, centroids_y, c='red', marker='X', s=400,
                  edgecolors='black', linewidths=3, zorder=5, label='Centroids')
        
        # Draw circles around centroids
        for k in range(n_clusters):
            circle = Circle((centroids_x[k], centroids_y[k]), 0.1, 
                           fill=False, color=colors[k], linewidth=2, linestyle='--')
            ax.add_patch(circle)
        
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Longitude (normalized)', fontsize=12)
        ax.set_ylabel('Latitude (normalized)', fontsize=12)
        ax.set_title(f'K-Means: Iteration {iteration + 1} - Assign Points to Nearest Centroid\n'
                    f'({len(X)} Knoxville Amenities)', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.text(0.5, -0.12, 'Each point assigned to nearest centroid (colored by cluster)', 
                ha='center', transform=ax.transAxes, fontsize=11, style='italic')
        
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        frames.append(img.copy())
        plt.close()
        
        # Update centroids
        new_centroids_x = np.array([lon_norm[labels == k].mean() if (labels == k).sum() > 0 
                                    else centroids_x[k] for k in range(n_clusters)])
        new_centroids_y = np.array([lat_norm[labels == k].mean() if (labels == k).sum() > 0 
                                    else centroids_y[k] for k in range(n_clusters)])
        
        # Frame: Show centroid movement
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        
        for k in range(n_clusters):
            cluster_mask = labels == k
            ax.scatter(lon_norm[cluster_mask], lat_norm[cluster_mask], 
                      c=colors[k], s=15, alpha=0.7)
        
        # Draw arrows from old to new centroids
        for k in range(n_clusters):
            ax.annotate('', xy=(new_centroids_x[k], new_centroids_y[k]),
                       xytext=(centroids_x[k], centroids_y[k]),
                       arrowprops=dict(arrowstyle='->', color='black', lw=2))
        
        ax.scatter(centroids_x, centroids_y, c='gray', marker='X', s=300,
                  edgecolors='black', linewidths=2, zorder=4, alpha=0.5, label='Old Centroids')
        ax.scatter(new_centroids_x, new_centroids_y, c='red', marker='X', s=400,
                  edgecolors='black', linewidths=3, zorder=5, label='New Centroids')
        
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Longitude (normalized)', fontsize=12)
        ax.set_ylabel('Latitude (normalized)', fontsize=12)
        ax.set_title(f'K-Means: Iteration {iteration + 1} - Move Centroids to Cluster Mean\n'
                    f'({len(X)} Knoxville Amenities)', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.text(0.5, -0.12, 'Centroids move to the center (mean) of their assigned points', 
                ha='center', transform=ax.transAxes, fontsize=11, style='italic')
        
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        frames.append(img.copy())
        plt.close()
        
        # Check convergence
        if np.allclose([centroids_x, centroids_y], [new_centroids_x, new_centroids_y], atol=0.001):
            print(f"  Converged at iteration {iteration + 1}")
            break
        
        centroids_x = new_centroids_x.copy()
        centroids_y = new_centroids_y.copy()
    
    # Final frame
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    
    for k in range(n_clusters):
        cluster_mask = labels == k
        count = cluster_mask.sum()
        ax.scatter(lon_norm[cluster_mask], lat_norm[cluster_mask], 
                  c=colors[k], s=20, alpha=0.8, label=f'Cluster {k} ({count} pts)')
    
    ax.scatter(centroids_x, centroids_y, c='red', marker='X', s=500,
              edgecolors='black', linewidths=3, zorder=5, label='Final Centroids')
    
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Longitude (normalized)', fontsize=12)
    ax.set_ylabel('Latitude (normalized)', fontsize=12)
    ax.set_title(f'K-Means: CONVERGED! Final Clusters\n'
                f'Silhouette Score: {kmeans_silhouette:.4f}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.text(0.5, -0.12, 'Algorithm converged - centroids no longer move!', 
            ha='center', transform=ax.transAxes, fontsize=11, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightgreen'))
    
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    frames.append(img.copy())
    plt.close()
    
    # Save as SLOW GIF (3 seconds per frame)
    imageio.mimsave('animation_kmeans_slow.gif', frames, duration=3.0, loop=0)
    print("✓ Saved: animation_kmeans_slow.gif (3 sec/frame)")
    
    # Also save as faster version
    imageio.mimsave('animation_kmeans_fast.gif', frames, duration=1.0, loop=0)
    print("✓ Saved: animation_kmeans_fast.gif (1 sec/frame)")

create_kmeans_slow_animation(X, lats, lons, n_clusters=5)

# =============================================================================
# 5. DBSCAN SLOW ANIMATION
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 5: Creating SLOW DBSCAN Animation")
print("=" * 70)

def create_dbscan_slow_animation(X, lats, lons, eps=0.05, min_samples=5):
    """Create SLOW DBSCAN animation for presentation"""
    
    print(f"\nCreating SLOW DBSCAN animation (3 sec/frame)...")
    
    # Normalize
    lat_norm = (lats - lats.min()) / (lats.max() - lats.min())
    lon_norm = (lons - lons.min()) / (lons.max() - lons.min())
    
    n_points = len(lat_norm)
    labels = np.full(n_points, -2)  # -2 = unvisited
    cluster_id = 0
    
    frames = []
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#bcbd22', '#17becf', '#aec7e8']
    
    fig_width, fig_height = 10, 8
    dpi = 100
    
    def get_neighbors(point_idx):
        distances = np.sqrt((lon_norm - lon_norm[point_idx]) ** 2 + 
                           (lat_norm - lat_norm[point_idx]) ** 2)
        return np.where(distances <= eps)[0]
    
    def save_frame(title, subtitle, current_point=None, neighbors=None, highlight_box=None):
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        
        # Unvisited
        unvisited_mask = labels == -2
        if unvisited_mask.sum() > 0:
            ax.scatter(lon_norm[unvisited_mask], lat_norm[unvisited_mask], 
                      c='lightgray', s=15, alpha=0.5, label='Unvisited')
        
        # Noise
        noise_mask = labels == -1
        if noise_mask.sum() > 0:
            ax.scatter(lon_norm[noise_mask], lat_norm[noise_mask],
                      c='black', s=30, marker='x', alpha=0.8, label=f'Noise ({noise_mask.sum()})')
        
        # Clustered
        for c in range(cluster_id + 1):
            cluster_mask = labels == c
            if cluster_mask.sum() > 0:
                ax.scatter(lon_norm[cluster_mask], lat_norm[cluster_mask],
                          c=colors[c % len(colors)], s=25, alpha=0.8, 
                          edgecolors='black', linewidths=0.3,
                          label=f'Cluster {c} ({cluster_mask.sum()})')
        
        # Current point
        if current_point is not None:
            ax.scatter(lon_norm[current_point], lat_norm[current_point],
                      c='red', s=200, marker='*', edgecolors='black', linewidths=2,
                      zorder=10, label='Current Point')
            
            circle = Circle((lon_norm[current_point], lat_norm[current_point]), eps, 
                           fill=False, color='red', linewidth=3, linestyle='--')
            ax.add_patch(circle)
            ax.text(lon_norm[current_point] + 0.02, lat_norm[current_point] + 0.02, 
                   f'ε={eps}', fontsize=10, color='red')
        
        # Neighbors
        if neighbors is not None and len(neighbors) > 0:
            ax.scatter(lon_norm[neighbors], lat_norm[neighbors],
                      c='yellow', s=60, marker='o', edgecolors='red', linewidths=2,
                      zorder=9, label=f'Neighbors ({len(neighbors)})')
        
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Longitude (normalized)', fontsize=12)
        ax.set_ylabel('Latitude (normalized)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        
        box_color = 'lightgreen' if highlight_box == 'green' else 'lightcoral' if highlight_box == 'red' else 'lightyellow'
        ax.text(0.5, -0.12, subtitle, ha='center', transform=ax.transAxes, 
                fontsize=11, style='italic', bbox=dict(boxstyle='round', facecolor=box_color))
        
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        frames.append(img.copy())
        plt.close()
    
    # Initial frame
    save_frame('DBSCAN: Initial State', 
               'All points are unvisited. Algorithm will check each point.',
               highlight_box='yellow')
    
    # Process sample points
    np.random.seed(42)
    sample_indices = np.random.choice(len(lon_norm), min(20, len(lon_norm)), replace=False)
    
    for idx in sample_indices:
        if labels[idx] != -2:
            continue
        
        neighbors = get_neighbors(idx)
        
        if len(neighbors) >= min_samples:
            # Core point
            labels[idx] = cluster_id
            save_frame(f'DBSCAN: CORE POINT Found! (Cluster {cluster_id})',
                      f'Neighbors: {len(neighbors)} >= minPts ({min_samples}) → Start new cluster!',
                      current_point=idx, neighbors=neighbors, highlight_box='green')
            
            # Expand cluster
            for neighbor in neighbors:
                if labels[neighbor] == -2:
                    labels[neighbor] = cluster_id
            
            save_frame(f'DBSCAN: Cluster {cluster_id} Expanded',
                      f'All {len(neighbors)} neighbors added to Cluster {cluster_id}',
                      highlight_box='green')
            
            cluster_id += 1
        else:
            # Noise point
            labels[idx] = -1
            save_frame('DBSCAN: NOISE Point Found',
                      f'Neighbors: {len(neighbors)} < minPts ({min_samples}) → Mark as noise',
                      current_point=idx, neighbors=neighbors, highlight_box='red')
    
    # Final frame
    save_frame(f'DBSCAN: COMPLETE!\n{cluster_id} Clusters, {(labels == -1).sum()} Noise Points',
               f'Silhouette Score: {dbscan_silhouette:.4f}',
               highlight_box='green')
    
    # Save as SLOW GIF
    imageio.mimsave('animation_dbscan_slow.gif', frames, duration=3.0, loop=0)
    print("✓ Saved: animation_dbscan_slow.gif (3 sec/frame)")
    
    imageio.mimsave('animation_dbscan_fast.gif', frames, duration=1.0, loop=0)
    print("✓ Saved: animation_dbscan_fast.gif (1 sec/frame)")

create_dbscan_slow_animation(X, lats, lons, eps=0.05, min_samples=5)

# =============================================================================
# 6. COMPARISON SLOW ANIMATION
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 6: Creating SLOW Comparison Animation")
print("=" * 70)

def create_comparison_slow_animation():
    """Create SLOW comparison animation"""
    
    print("\nCreating SLOW comparison animation (4 sec/frame)...")
    
    lat_norm = (lats - lats.min()) / (lats.max() - lats.min())
    lon_norm = (lons - lons.min()) / (lons.max() - lons.min())
    
    frames = []
    colors_km = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    colors_db = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#bcbd22', '#17becf', '#aec7e8']
    
    fig_width, fig_height = 10, 8
    dpi = 100
    
    # Frame 1: Original
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax.scatter(lon_norm, lat_norm, c='gray', s=20, alpha=0.6)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'ORIGINAL DATA\n{len(X)} Knoxville Amenities (restaurants, churches, schools...)', 
                fontsize=14, fontweight='bold')
    ax.text(0.5, -0.1, 'Raw data points - no clustering applied yet', 
            ha='center', transform=ax.transAxes, fontsize=11, style='italic')
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    frames.append(img.copy())
    plt.close()
    
    # Frame 2: K-Means
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    for k in range(K_OPTIMAL):
        mask = kmeans_labels == k
        ax.scatter(lon_norm[mask], lat_norm[mask], c=colors_km[k], s=20, alpha=0.7,
                  label=f'Cluster {k} ({mask.sum()})')
    
    # Centroids in lat/lon space
    centroids_lon = (kmeans_centroids[:, 0] - X[:, 0].min()) / (X[:, 0].max() - X[:, 0].min())
    centroids_lat = (kmeans_centroids[:, 1] - X[:, 1].min()) / (X[:, 1].max() - X[:, 1].min())
    ax.scatter(centroids_lon, centroids_lat, c='red', marker='X', s=300,
              edgecolors='black', linewidths=2, zorder=5, label='Centroids')
    
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'K-MEANS CLUSTERING (K=5)\nSilhouette: {kmeans_silhouette:.4f}', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.text(0.5, -0.1, 'All points assigned to exactly 5 clusters', 
            ha='center', transform=ax.transAxes, fontsize=11, style='italic')
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    frames.append(img.copy())
    plt.close()
    
    # Frame 3: DBSCAN
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    
    # Noise first
    noise_mask = dbscan_labels == -1
    if noise_mask.sum() > 0:
        ax.scatter(lon_norm[noise_mask], lat_norm[noise_mask], c='black', s=15, 
                  marker='x', alpha=0.6, label=f'Noise ({noise_mask.sum()})')
    
    # Clusters
    for c in range(min(n_clusters_dbscan, 10)):
        mask = dbscan_labels == c
        if mask.sum() > 0:
            ax.scatter(lon_norm[mask], lat_norm[mask], c=colors_db[c], s=20, alpha=0.7,
                      label=f'Cluster {c} ({mask.sum()})')
    
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'DBSCAN CLUSTERING\n{n_clusters_dbscan} clusters, {n_noise} noise | Silhouette: {dbscan_silhouette:.4f}', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.text(0.5, -0.1, f'Found {n_clusters_dbscan} natural clusters + {n_noise} outliers (black X)', 
            ha='center', transform=ax.transAxes, fontsize=11, style='italic')
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    frames.append(img.copy())
    plt.close()
    
    # Frame 4: Winner
    winner = "DBSCAN" if dbscan_silhouette > kmeans_silhouette else "K-Means"
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    
    ax.text(0.5, 0.7, f'🏆 WINNER: {winner}', ha='center', va='center',
            fontsize=36, fontweight='bold', transform=ax.transAxes)
    
    ax.text(0.5, 0.5, f'K-Means Silhouette: {kmeans_silhouette:.4f}\n'
                      f'DBSCAN Silhouette: {dbscan_silhouette:.4f}', 
            ha='center', va='center', fontsize=18, transform=ax.transAxes)
    
    ax.text(0.5, 0.25, f'DBSCAN found {n_noise} outliers that K-Means forced into clusters',
            ha='center', va='center', fontsize=14, transform=ax.transAxes, style='italic')
    
    ax.axis('off')
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    frames.append(img.copy())
    plt.close()
    
    # Save
    imageio.mimsave('animation_comparison_slow.gif', frames, duration=4.0, loop=0)
    print("✓ Saved: animation_comparison_slow.gif (4 sec/frame)")
    
    imageio.mimsave('animation_comparison_fast.gif', frames, duration=1.5, loop=0)
    print("✓ Saved: animation_comparison_fast.gif (1.5 sec/frame)")

create_comparison_slow_animation()

# =============================================================================
# 7. STATIC IMAGES
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 7: Creating Static Images")
print("=" * 70)

# K-Means steps
print("\nCreating K-Means steps image...")
# (Already created in previous code - keeping this section for completeness)

# DBSCAN concept
def create_dbscan_concept():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    np.random.seed(42)
    
    # Core
    ax = axes[0]
    center = np.array([0.5, 0.5])
    neighbors = center + np.random.randn(8, 2) * 0.1
    ax.scatter(neighbors[:, 0], neighbors[:, 1], c='blue', s=150, alpha=0.7, 
              edgecolors='black', linewidths=1, label='8 Neighbors')
    ax.scatter(center[0], center[1], c='red', s=300, marker='*', 
              edgecolors='black', linewidths=2, label='Core Point', zorder=5)
    circle = Circle(center, 0.15, fill=False, color='red', linewidth=3, linestyle='--')
    ax.add_patch(circle)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title('CORE POINT\n8 neighbors >= minPts(5) ✓', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_aspect('equal')
    
    # Border
    ax = axes[1]
    core = np.array([0.4, 0.5])
    border = np.array([0.6, 0.5])
    neighbors = core + np.random.randn(6, 2) * 0.08
    ax.scatter(neighbors[:, 0], neighbors[:, 1], c='blue', s=150, alpha=0.7, edgecolors='black')
    ax.scatter(core[0], core[1], c='red', s=300, marker='*', edgecolors='black', linewidths=2, label='Core', zorder=5)
    ax.scatter(border[0], border[1], c='orange', s=250, marker='s', edgecolors='black', linewidths=2, label='Border', zorder=5)
    circle1 = Circle(core, 0.15, fill=False, color='red', linewidth=3, linestyle='--')
    circle2 = Circle(border, 0.15, fill=False, color='orange', linewidth=3, linestyle='--')
    ax.add_patch(circle1); ax.add_patch(circle2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title('BORDER POINT\n< minPts, but in Core\'s ε', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_aspect('equal')
    
    # Noise
    ax = axes[2]
    cluster = np.array([[0.3, 0.5], [0.35, 0.55], [0.4, 0.45], [0.32, 0.48], [0.38, 0.52]])
    noise = np.array([0.8, 0.8])
    ax.scatter(cluster[:, 0], cluster[:, 1], c='blue', s=150, alpha=0.7, edgecolors='black', label='Cluster')
    ax.scatter(noise[0], noise[1], c='black', s=300, marker='X', linewidths=3, label='Noise', zorder=5)
    circle = Circle(noise, 0.15, fill=False, color='black', linewidth=3, linestyle='--')
    ax.add_patch(circle)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title('NOISE POINT\n< minPts, isolated ✗', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_aspect('equal')
    
    plt.suptitle('DBSCAN Point Classification (minPts=5, ε=0.15)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('dbscan_concept.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: dbscan_concept.png")

create_dbscan_concept()

# Comparison final
print("\nCreating final comparison image...")
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

lat_norm = (lats - lats.min()) / (lats.max() - lats.min())
lon_norm = (lons - lons.min()) / (lons.max() - lons.min())

colors_km = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
colors_db = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf']

# Original
axes[0].scatter(lon_norm, lat_norm, c='gray', s=15, alpha=0.6)
axes[0].set_title(f'Original Data\n({len(X)} Amenities)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')

# K-Means
for k in range(K_OPTIMAL):
    mask = kmeans_labels == k
    axes[1].scatter(lon_norm[mask], lat_norm[mask], c=colors_km[k], s=15, alpha=0.7, label=f'C{k}')
axes[1].set_title(f'K-Means (K=5)\nSilhouette: {kmeans_silhouette:.4f}', fontsize=14, fontweight='bold')
axes[1].legend(loc='upper right', fontsize=8)
axes[1].set_xlabel('Longitude')

# DBSCAN
noise_mask = dbscan_labels == -1
axes[2].scatter(lon_norm[noise_mask], lat_norm[noise_mask], c='black', s=10, marker='x', alpha=0.5, label=f'Noise ({n_noise})')
for c in range(min(n_clusters_dbscan, 9)):
    mask = dbscan_labels == c
    if mask.sum() > 0:
        axes[2].scatter(lon_norm[mask], lat_norm[mask], c=colors_db[c], s=15, alpha=0.7, label=f'C{c}')
axes[2].set_title(f'DBSCAN\nSilhouette: {dbscan_silhouette:.4f}, Noise: {n_noise}', fontsize=14, fontweight='bold')
axes[2].legend(loc='upper right', fontsize=7, ncol=2)
axes[2].set_xlabel('Longitude')

plt.tight_layout()
plt.savefig('comparison_final.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: comparison_final.png")

# =============================================================================
# 8. INTERACTIVE MAPS
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 8: Creating Interactive Maps")
print("=" * 70)

import folium
from folium.plugins import HeatMap, MiniMap, DualMap
from pyproj import Transformer

center = [lats.mean(), lons.mean()]

# 8.1 All amenities
print("\n8.1 Creating All Amenities Map...")
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
        location=[row.geometry.y, row.geometry.x], radius=5,
        color=color, fill=True, fillColor=color, fillOpacity=0.8,
        popup=f"<b>{amenity}</b>"
    ).add_to(m_all)

m_all.save("map_all_amenities.html")
print("✓ Saved: map_all_amenities.html")

# 8.2 Heatmap
print("\n8.2 Creating Heatmap...")
m_heat = folium.Map(location=center, zoom_start=12, tiles='CartoDB dark_matter')
heat_data = [[row.geometry.y, row.geometry.x] for idx, row in gdf_latlon.iterrows()]
HeatMap(heat_data, radius=15, blur=10).add_to(m_heat)
m_heat.save("map_heatmap.html")
print("✓ Saved: map_heatmap.html")

# 8.3 K-Means Map
print("\n8.3 Creating K-Means Map...")
m_km = folium.Map(location=center, zoom_start=12)
km_colors_hex = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

for idx, row in gdf_latlon.iterrows():
    cluster = gdf_projected.loc[idx, "kmeans_cluster"]
    amenity = gdf_projected.loc[idx, "amenity"]
    color = km_colors_hex[cluster % len(km_colors_hex)]
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x], radius=5,
        color=color, fill=True, fillColor=color, fillOpacity=0.8,
        popup=f"<b>{amenity}</b><br>Cluster: {cluster}"
    ).add_to(m_km)

# Add centroids
transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
for i, centroid in enumerate(kmeans_centroids):
    lon, lat = transformer.transform(centroid[0], centroid[1])
    folium.Marker(
        location=[lat, lon], popup=f"<b>Centroid {i}</b>",
        icon=folium.Icon(color='red', icon='star')
    ).add_to(m_km)

m_km.save("map_kmeans.html")
print("✓ Saved: map_kmeans.html")

# 8.4 DBSCAN Map
print("\n8.4 Creating DBSCAN Map...")
m_db = folium.Map(location=center, zoom_start=12)
db_colors_hex = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

for idx, row in gdf_latlon.iterrows():
    cluster = gdf_projected.loc[idx, "dbscan_cluster"]
    amenity = gdf_projected.loc[idx, "amenity"]
    if cluster == -1:
        color = 'black'
        radius = 4
    else:
        color = db_colors_hex[cluster % len(db_colors_hex)]
        radius = 6
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x], radius=radius,
        color=color, fill=True, fillColor=color, fillOpacity=0.8,
        popup=f"<b>{amenity}</b><br>{'NOISE' if cluster == -1 else f'Cluster: {cluster}'}"
    ).add_to(m_db)

m_db.save("map_dbscan.html")
print("✓ Saved: map_dbscan.html")

# 8.5 Side-by-Side
print("\n8.5 Creating Side-by-Side Map...")
m_dual = DualMap(location=center, zoom_start=12)

for idx, row in gdf_latlon.iterrows():
    cluster = gdf_projected.loc[idx, "kmeans_cluster"]
    color = km_colors_hex[cluster % len(km_colors_hex)]
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x], radius=4,
        color=color, fill=True, fillColor=color, fillOpacity=0.8
    ).add_to(m_dual.m1)

for idx, row in gdf_latlon.iterrows():
    cluster = gdf_projected.loc[idx, "dbscan_cluster"]
    color = 'black' if cluster == -1 else db_colors_hex[cluster % len(db_colors_hex)]
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x], radius=4,
        color=color, fill=True, fillColor=color, fillOpacity=0.8
    ).add_to(m_dual.m2)

m_dual.save("map_comparison_dual.html")
print("✓ Saved: map_comparison_dual.html")

# =============================================================================
# 9. FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

winner = "DBSCAN" if dbscan_silhouette > kmeans_silhouette else "K-Means"

print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║          K-MEANS vs DBSCAN: KNOXVILLE AMENITIES                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Dataset: {len(X)} amenities from OpenStreetMap                            ║
║                                                                          ║
║  K-MEANS (K=5)           DBSCAN (eps={EPS_OPTIMAL}, minPts={MIN_SAMPLES})               ║
║  Silhouette: {kmeans_silhouette:.4f}        Silhouette: {dbscan_silhouette:.4f}                       ║
║  Clusters: 5              Clusters: {n_clusters_dbscan}                              ║
║  Noise: 0                 Noise: {n_noise}                                 ║
║                                                                          ║
║  🏆 WINNER: {winner}                                                      ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print("""
SLOW ANIMATIONS (for presentation - 3-4 sec/frame):
  ✓ animation_kmeans_slow.gif      - K-Means step-by-step (3 sec/frame)
  ✓ animation_dbscan_slow.gif      - DBSCAN step-by-step (3 sec/frame)
  ✓ animation_comparison_slow.gif  - Comparison (4 sec/frame)

FAST ANIMATIONS (for quick preview):
  ✓ animation_kmeans_fast.gif      - K-Means (1 sec/frame)
  ✓ animation_dbscan_fast.gif      - DBSCAN (1 sec/frame)
  ✓ animation_comparison_fast.gif  - Comparison (1.5 sec/frame)

STATIC IMAGES:
  ✓ dbscan_concept.png             - Core/Border/Noise illustration
  ✓ comparison_final.png           - Side-by-side comparison

INTERACTIVE MAPS (open in browser):
  ✓ map_all_amenities.html         - All amenities by type
  ✓ map_heatmap.html               - Density heatmap
  ✓ map_kmeans.html                - K-Means on Knoxville map
  ✓ map_dbscan.html                - DBSCAN on Knoxville map
  ✓ map_comparison_dual.html       - Side-by-side K-Means vs DBSCAN
""")

print("\n" + "=" * 70)
print("SCRIPT COMPLETED SUCCESSFULLY! ✓")
print("=" * 70)
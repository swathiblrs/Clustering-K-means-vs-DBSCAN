"""
================================================================================
K-Means vs DBSCAN: Earthquake Epicenter Clustering
================================================================================
COSC 527 - Biologically-Inspired Computation
Group Presentation - April 23, 2026

Dataset: USGS Earthquake Catalog (2020-2024, Magnitude 5.0+)
Features: latitude, longitude, depth, magnitude
================================================================================
"""

# =============================================================================
# 1. IMPORT LIBRARIES
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

print("="*70)
print("K-Means vs DBSCAN: Earthquake Epicenter Clustering")
print("="*70)

# =============================================================================
# 2. LOAD AND EXPLORE DATA
# =============================================================================
print("\n" + "="*70)
print("SECTION 2: Loading and Exploring Data")
print("="*70)

# Load dataset
df = pd.read_csv('query.csv')

print(f"\nDataset Shape: {df.shape}")
print(f"Total Earthquakes: {len(df):,}")
print(f"\nColumns Available:")
print(df.columns.tolist())

print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nData Types:")
print(df.dtypes)

print(f"\nStatistical Summary of Key Features:")
print(df[['latitude', 'longitude', 'depth', 'mag']].describe())

# =============================================================================
# 3. DATA PREPROCESSING
# =============================================================================
print("\n" + "="*70)
print("SECTION 3: Data Preprocessing")
print("="*70)

# Select features for clustering
features = ['latitude', 'longitude', 'depth', 'mag']

# Create a copy with selected features
df_cluster = df[features].copy()

# Check for missing values
print(f"\nMissing Values:")
print(df_cluster.isnull().sum())

# Drop rows with missing values
df_cluster = df_cluster.dropna()
print(f"\nRecords after dropping NaN: {len(df_cluster):,}")

# Store original data for visualization
X_original = df_cluster.values

# Normalize features using StandardScaler
# This is IMPORTANT because features have different scales:
# - latitude: -90 to 90
# - longitude: -180 to 180
# - depth: 0 to 700+ km
# - magnitude: 5.0 to 9.0+
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster)

print(f"\nScaled Data Shape: {X_scaled.shape}")
print(f"\nAfter Scaling:")
print(f"  Mean of each feature: {X_scaled.mean(axis=0).round(4)}")
print(f"  Std of each feature: {X_scaled.std(axis=0).round(4)}")

# =============================================================================
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
print("\n" + "="*70)
print("SECTION 4: Exploratory Data Analysis")
print("="*70)

# 4.1 Visualize earthquake distribution on world map
print("\nGenerating earthquake distribution map...")
fig, ax = plt.subplots(figsize=(14, 8))

scatter = ax.scatter(
    df_cluster['longitude'], 
    df_cluster['latitude'],
    c=df_cluster['mag'],
    cmap='YlOrRd',
    s=df_cluster['mag']**2,  # Size by magnitude
    alpha=0.5,
    edgecolors='none'
)

plt.colorbar(scatter, label='Magnitude')
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_title('Global Earthquake Distribution (2020-2024, Mag 5.0+)', fontsize=14)
ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('01_earthquake_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 01_earthquake_distribution.png")

# 4.2 Distribution of features
print("\nGenerating feature distribution plots...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Latitude distribution
axes[0, 0].hist(df_cluster['latitude'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Latitude')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of Latitude')

# Longitude distribution
axes[0, 1].hist(df_cluster['longitude'], bins=50, color='seagreen', edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Longitude')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Distribution of Longitude')

# Depth distribution
axes[1, 0].hist(df_cluster['depth'], bins=50, color='coral', edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel('Depth (km)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Distribution of Depth')

# Magnitude distribution
axes[1, 1].hist(df_cluster['mag'], bins=30, color='orchid', edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('Magnitude')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Distribution of Magnitude')

plt.tight_layout()
plt.savefig('02_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 02_feature_distributions.png")

# 4.3 Correlation heatmap
print("\nGenerating correlation heatmap...")
plt.figure(figsize=(8, 6))
correlation = df_cluster.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, fmt='.2f')
plt.title('Feature Correlation Matrix', fontsize=14)
plt.tight_layout()
plt.savefig('03_correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 03_correlation_matrix.png")

# =============================================================================
# 5. K-MEANS CLUSTERING
# =============================================================================
print("\n" + "="*70)
print("SECTION 5: K-Means Clustering")
print("="*70)

# 5.1 Finding Optimal K using Elbow Method and Silhouette Score
print("\n5.1 Finding Optimal K...")
print("-"*50)

K_range = range(2, 15)
inertias = []
silhouette_scores_kmeans = []

print(f"\n{'K':<5} {'Inertia':<15} {'Silhouette Score':<15}")
print("-"*40)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    sil_score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores_kmeans.append(sil_score)
    print(f"{k:<5} {kmeans.inertia_:<15.2f} {sil_score:<15.4f}")

# Find optimal K based on silhouette score
optimal_k_silhouette = K_range[np.argmax(silhouette_scores_kmeans)]
print(f"\nOptimal K based on Silhouette Score: {optimal_k_silhouette}")

# 5.2 Plot Elbow Curve and Silhouette Scores
print("\nGenerating Elbow and Silhouette plots...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow curve
axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[0].set_ylabel('Inertia (Within-cluster Sum of Squares)', fontsize=12)
axes[0].set_title('Elbow Method for Optimal K', fontsize=14)
axes[0].set_xticks(list(K_range))
axes[0].grid(True, alpha=0.3)

# Silhouette scores
axes[1].plot(K_range, silhouette_scores_kmeans, 'go-', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[1].set_ylabel('Silhouette Score', fontsize=12)
axes[1].set_title('Silhouette Score vs K', fontsize=14)
axes[1].set_xticks(list(K_range))
axes[1].grid(True, alpha=0.3)
# Highlight optimal K
axes[1].axvline(x=optimal_k_silhouette, color='red', linestyle='--', label=f'Optimal K={optimal_k_silhouette}')
axes[1].legend()

plt.tight_layout()
plt.savefig('04_kmeans_elbow_silhouette.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 04_kmeans_elbow_silhouette.png")

# 5.3 Apply K-Means with Optimal K
print("\n5.2 Applying K-Means with Optimal K...")
print("-"*50)

# You can adjust this based on elbow/silhouette analysis
K_OPTIMAL = 4  # Using 7 as a good balance

print(f"\nUsing K = {K_OPTIMAL}")

# Fit K-Means
kmeans_final = KMeans(n_clusters=K_OPTIMAL, random_state=42, n_init=10)
kmeans_labels = kmeans_final.fit_predict(X_scaled)

# Add labels to dataframe
df_cluster['kmeans_cluster'] = kmeans_labels

# Display cluster distribution
print(f"\nK-Means Cluster Distribution:")
print(df_cluster['kmeans_cluster'].value_counts().sort_index())

# 5.4 K-Means Evaluation Metrics
kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
kmeans_calinski = calinski_harabasz_score(X_scaled, kmeans_labels)
kmeans_davies = davies_bouldin_score(X_scaled, kmeans_labels)

print(f"\nK-Means Evaluation Metrics (K={K_OPTIMAL}):")
print(f"  Silhouette Score: {kmeans_silhouette:.4f} (higher is better, range: -1 to 1)")
print(f"  Calinski-Harabasz Index: {kmeans_calinski:.2f} (higher is better)")
print(f"  Davies-Bouldin Index: {kmeans_davies:.4f} (lower is better)")

# 5.5 Visualize K-Means clusters on map
print("\nGenerating K-Means cluster map...")
fig, ax = plt.subplots(figsize=(14, 8))

scatter = ax.scatter(
    df_cluster['longitude'], 
    df_cluster['latitude'],
    c=df_cluster['kmeans_cluster'],
    cmap='tab10',
    s=15,
    alpha=0.6,
    edgecolors='none'
)

# Plot cluster centers
centers_original = scaler.inverse_transform(kmeans_final.cluster_centers_)
ax.scatter(
    centers_original[:, 1],  # longitude
    centers_original[:, 0],  # latitude
    c='red', marker='X', s=200, edgecolors='black', linewidths=2,
    label='Cluster Centers'
)

plt.colorbar(scatter, label='Cluster')
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_title(f'K-Means Clustering (K={K_OPTIMAL})', fontsize=14)
ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.legend(loc='lower left')

plt.tight_layout()
plt.savefig('05_kmeans_clusters_map.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 05_kmeans_clusters_map.png")

# =============================================================================
# 6. DBSCAN CLUSTERING
# =============================================================================
print("\n" + "="*70)
print("SECTION 6: DBSCAN Clustering")
print("="*70)

# 6.1 Finding Optimal eps using k-distance graph
print("\n6.1 Finding Optimal eps using K-Distance Graph...")
print("-"*50)

# Rule of thumb: min_samples = 2 * n_features = 2 * 4 = 8
min_samples = 8
print(f"Using min_samples = {min_samples} (rule of thumb: 2 * n_features)")

# Find k-nearest neighbors distances
neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)

# Sort distances to k-th nearest neighbor
k_distances = np.sort(distances[:, min_samples-1])

# Plot k-distance graph
print("\nGenerating K-Distance graph...")
plt.figure(figsize=(10, 6))
plt.plot(range(len(k_distances)), k_distances, 'b-', linewidth=1)
plt.xlabel('Points (sorted by distance)', fontsize=12)
plt.ylabel(f'{min_samples}-th Nearest Neighbor Distance', fontsize=12)
plt.title('K-Distance Graph for eps Selection', fontsize=14)
plt.grid(True, alpha=0.3)

# Add horizontal lines for reference
for eps_val in [0.3, 0.4, 0.5, 0.6]:
    plt.axhline(y=eps_val, color='red', linestyle='--', alpha=0.5, label=f'eps={eps_val}')

plt.legend()
plt.tight_layout()
plt.savefig('06_dbscan_kdistance.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 06_dbscan_kdistance.png")

print(f"\nPercentile distances:")
print(f"  90th percentile: {np.percentile(k_distances, 90):.3f}")
print(f"  95th percentile: {np.percentile(k_distances, 95):.3f}")
print(f"  99th percentile: {np.percentile(k_distances, 99):.3f}")

# 6.2 Test Different eps Values
print("\n6.2 Testing Different eps Values...")
print("-"*50)

eps_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

print(f"\n{'eps':<8} {'Clusters':<10} {'Noise':<10} {'Noise %':<10} {'Silhouette':<12}")
print("-"*55)

dbscan_results = []

for eps in eps_values:
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    noise_pct = (n_noise / len(labels)) * 100
    
    # Calculate silhouette only if we have valid clusters
    if n_clusters > 1 and n_noise < len(labels) - 1:
        mask = labels != -1
        if sum(mask) > n_clusters:
            sil_score = silhouette_score(X_scaled[mask], labels[mask])
        else:
            sil_score = -1
    else:
        sil_score = -1
    
    dbscan_results.append({
        'eps': eps,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'noise_pct': noise_pct,
        'silhouette': sil_score
    })
    
    print(f"{eps:<8} {n_clusters:<10} {n_noise:<10} {noise_pct:<10.2f} {sil_score:<12.4f}")

# 6.3 Apply DBSCAN with Optimal Parameters
print("\n6.3 Applying DBSCAN with Optimal Parameters...")
print("-"*50)

# Choose optimal eps (you can adjust based on above results)
EPS_OPTIMAL = 0.4
MIN_SAMPLES = 10

print(f"\nUsing eps = {EPS_OPTIMAL}, min_samples = {MIN_SAMPLES}")

# Fit DBSCAN
dbscan_final = DBSCAN(eps=EPS_OPTIMAL, min_samples=MIN_SAMPLES)
dbscan_labels = dbscan_final.fit_predict(X_scaled)

# Add labels to dataframe
df_cluster['dbscan_cluster'] = dbscan_labels

# Calculate statistics
n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

print(f"\nDBSCAN Results:")
print(f"  Number of clusters: {n_clusters_dbscan}")
print(f"  Number of noise points: {n_noise} ({(n_noise/len(dbscan_labels))*100:.2f}%)")
print(f"\nCluster Distribution:")
print(df_cluster['dbscan_cluster'].value_counts().sort_index())

# 6.4 DBSCAN Evaluation Metrics
mask = dbscan_labels != -1
if sum(mask) > n_clusters_dbscan and n_clusters_dbscan > 1:
    dbscan_silhouette = silhouette_score(X_scaled[mask], dbscan_labels[mask])
    dbscan_calinski = calinski_harabasz_score(X_scaled[mask], dbscan_labels[mask])
    dbscan_davies = davies_bouldin_score(X_scaled[mask], dbscan_labels[mask])
    
    print(f"\nDBSCAN Evaluation Metrics (excluding noise):")
    print(f"  Silhouette Score: {dbscan_silhouette:.4f} (higher is better)")
    print(f"  Calinski-Harabasz Index: {dbscan_calinski:.2f} (higher is better)")
    print(f"  Davies-Bouldin Index: {dbscan_davies:.4f} (lower is better)")
else:
    print("\nNot enough clusters to calculate metrics.")
    dbscan_silhouette = -1
    dbscan_calinski = 0
    dbscan_davies = float('inf')

# 6.5 Visualize DBSCAN clusters on map
print("\nGenerating DBSCAN cluster map...")
fig, ax = plt.subplots(figsize=(14, 8))

# Separate noise and clustered points
noise_mask = df_cluster['dbscan_cluster'] == -1
clustered_mask = ~noise_mask

# Plot clustered points
scatter = ax.scatter(
    df_cluster.loc[clustered_mask, 'longitude'], 
    df_cluster.loc[clustered_mask, 'latitude'],
    c=df_cluster.loc[clustered_mask, 'dbscan_cluster'],
    cmap='tab10',
    s=15,
    alpha=0.6,
    edgecolors='none',
    label='Clustered'
)

# Plot noise points
ax.scatter(
    df_cluster.loc[noise_mask, 'longitude'], 
    df_cluster.loc[noise_mask, 'latitude'],
    c='gray',
    s=10,
    alpha=0.3,
    marker='x',
    label=f'Noise ({n_noise} points)'
)

plt.colorbar(scatter, label='Cluster')
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_title(f'DBSCAN Clustering (eps={EPS_OPTIMAL}, min_samples={MIN_SAMPLES})', fontsize=14)
ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.legend(loc='lower left')

plt.tight_layout()
plt.savefig('07_dbscan_clusters_map.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 07_dbscan_clusters_map.png")

# =============================================================================
# 7. SIDE-BY-SIDE COMPARISON
# =============================================================================
print("\n" + "="*70)
print("SECTION 7: Side-by-Side Comparison")
print("="*70)

# 7.1 Side-by-side visualization
print("\nGenerating side-by-side comparison map...")
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# K-Means
scatter1 = axes[0].scatter(
    df_cluster['longitude'], 
    df_cluster['latitude'],
    c=df_cluster['kmeans_cluster'],
    cmap='tab10',
    s=10,
    alpha=0.6
)
axes[0].set_xlabel('Longitude', fontsize=12)
axes[0].set_ylabel('Latitude', fontsize=12)
axes[0].set_title(f'K-Means (K={K_OPTIMAL})\nAll points assigned to clusters', fontsize=13)
axes[0].set_xlim(-180, 180)
axes[0].set_ylim(-90, 90)
plt.colorbar(scatter1, ax=axes[0], label='Cluster')

# DBSCAN
scatter2 = axes[1].scatter(
    df_cluster.loc[clustered_mask, 'longitude'], 
    df_cluster.loc[clustered_mask, 'latitude'],
    c=df_cluster.loc[clustered_mask, 'dbscan_cluster'],
    cmap='tab10',
    s=10,
    alpha=0.6
)
axes[1].scatter(
    df_cluster.loc[noise_mask, 'longitude'], 
    df_cluster.loc[noise_mask, 'latitude'],
    c='lightgray',
    s=5,
    alpha=0.3,
    label='Noise'
)
axes[1].set_xlabel('Longitude', fontsize=12)
axes[1].set_ylabel('Latitude', fontsize=12)
axes[1].set_title(f'DBSCAN (eps={EPS_OPTIMAL}, min_samples={MIN_SAMPLES})\n{n_clusters_dbscan} clusters + {n_noise} noise points', fontsize=13)
axes[1].set_xlim(-180, 180)
axes[1].set_ylim(-90, 90)
axes[1].legend(loc='lower left')
plt.colorbar(scatter2, ax=axes[1], label='Cluster')

plt.tight_layout()
plt.savefig('08_comparison_side_by_side.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 08_comparison_side_by_side.png")

# 7.2 Comparison metrics table
print("\n" + "="*70)
print("COMPARISON TABLE: K-Means vs DBSCAN")
print("="*70)

comparison_data = {
    'Metric': [
        'Number of Clusters',
        'Noise Points',
        'Silhouette Score',
        'Calinski-Harabasz Index',
        'Davies-Bouldin Index',
        'Requires K?',
        'Handles Outliers?',
        'Cluster Shape'
    ],
    'K-Means': [
        str(K_OPTIMAL),
        '0 (assigns all)',
        f'{kmeans_silhouette:.4f}',
        f'{kmeans_calinski:.2f}',
        f'{kmeans_davies:.4f}',
        'Yes',
        'No',
        'Spherical/Convex'
    ],
    'DBSCAN': [
        str(n_clusters_dbscan),
        f'{n_noise} ({(n_noise/len(dbscan_labels))*100:.1f}%)',
        f'{dbscan_silhouette:.4f}' if dbscan_silhouette != -1 else 'N/A',
        f'{dbscan_calinski:.2f}' if dbscan_calinski != 0 else 'N/A',
        f'{dbscan_davies:.4f}' if dbscan_davies != float('inf') else 'N/A',
        'No (uses eps, min_samples)',
        'Yes',
        'Arbitrary'
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))

# =============================================================================
# 8. CLUSTER ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("SECTION 8: Cluster Analysis")
print("="*70)

# 8.1 K-Means cluster statistics
print("\n8.1 K-Means Cluster Statistics:")
print("-"*60)
kmeans_stats = df_cluster.groupby('kmeans_cluster')[features].agg(['mean', 'std', 'count'])
print(kmeans_stats.round(2))

# 8.2 DBSCAN cluster statistics
print("\n8.2 DBSCAN Cluster Statistics (excluding noise):")
print("-"*60)
dbscan_valid = df_cluster[df_cluster['dbscan_cluster'] != -1]
if len(dbscan_valid) > 0:
    dbscan_stats = dbscan_valid.groupby('dbscan_cluster')[features].agg(['mean', 'std', 'count'])
    print(dbscan_stats.round(2))

# 8.3 Visualize cluster characteristics
print("\nGenerating cluster characteristics plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# K-Means: Depth by cluster
df_cluster.boxplot(column='depth', by='kmeans_cluster', ax=axes[0, 0])
axes[0, 0].set_title('K-Means: Depth Distribution by Cluster')
axes[0, 0].set_xlabel('Cluster')
axes[0, 0].set_ylabel('Depth (km)')

# K-Means: Magnitude by cluster
df_cluster.boxplot(column='mag', by='kmeans_cluster', ax=axes[0, 1])
axes[0, 1].set_title('K-Means: Magnitude Distribution by Cluster')
axes[0, 1].set_xlabel('Cluster')
axes[0, 1].set_ylabel('Magnitude')

# DBSCAN: Depth by cluster
if len(dbscan_valid) > 0:
    dbscan_valid.boxplot(column='depth', by='dbscan_cluster', ax=axes[1, 0])
    axes[1, 0].set_title('DBSCAN: Depth Distribution by Cluster')
    axes[1, 0].set_xlabel('Cluster')
    axes[1, 0].set_ylabel('Depth (km)')

    # DBSCAN: Magnitude by cluster
    dbscan_valid.boxplot(column='mag', by='dbscan_cluster', ax=axes[1, 1])
    axes[1, 1].set_title('DBSCAN: Magnitude Distribution by Cluster')
    axes[1, 1].set_xlabel('Cluster')
    axes[1, 1].set_ylabel('Magnitude')

plt.suptitle('')
plt.tight_layout()
plt.savefig('09_cluster_characteristics.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 09_cluster_characteristics.png")

# =============================================================================
# 9. KEY FINDINGS AND CONCLUSIONS
# =============================================================================
print("\n" + "="*70)
print("SECTION 9: Key Findings and Conclusions")
print("="*70)

print("""
================================================================================
KEY FINDINGS: K-Means vs DBSCAN for Earthquake Epicenter Clustering
================================================================================

1. CLUSTER FORMATION:
   - K-Means: Creates exactly K spherical clusters, assigns ALL points
   - DBSCAN: Creates variable clusters based on density, identifies outliers

2. OUTLIER HANDLING:
   - K-Means: Forces isolated earthquakes into nearest cluster (may distort)
   - DBSCAN: Correctly identifies isolated earthquakes as noise/anomalies

3. GEOGRAPHIC PATTERNS:
   - K-Means: May split natural seismic zones or merge different regions
   - DBSCAN: Better follows natural fault lines and tectonic boundaries

4. PARAMETER SENSITIVITY:
   - K-Means: Requires choosing K (number of clusters) beforehand
   - DBSCAN: Requires eps (radius) and min_samples, but no fixed cluster count

5. BEST USE CASES:
   - K-Means: When you need a fixed number of regions, spherical clusters expected
   - DBSCAN: When density varies, outlier detection important, irregular shapes

6. FOR EARTHQUAKE DATA:
   - DBSCAN is generally more suitable because:
     * Earthquake clusters follow fault lines (irregular shapes)
     * Isolated earthquakes should be identified as anomalies
     * Density varies significantly across regions

================================================================================
""")

# =============================================================================
# 10. SAVE RESULTS
# =============================================================================
print("\n" + "="*70)
print("SECTION 10: Saving Results")
print("="*70)

# Save clustered data to CSV
df_cluster.to_csv('earthquake_clustering_results.csv', index=False)
print("Saved: earthquake_clustering_results.csv")

# Save comparison table
comparison_df.to_csv('algorithm_comparison.csv', index=False)
print("Saved: algorithm_comparison.csv")

# =============================================================================
# 11. SUMMARY TABLE FOR PRESENTATION
# =============================================================================
print("\n" + "="*70)
print("SECTION 11: Summary Table for Presentation")
print("="*70)

summary_table = """
+---------------------------+--------------------------------+--------------------------------+
|         Aspect            |           K-Means              |            DBSCAN              |
+---------------------------+--------------------------------+--------------------------------+
| Requires K?               | Yes                            | No                             |
| Parameters                | K (number of clusters)         | eps, min_samples               |
| Cluster Shape             | Spherical/Convex               | Arbitrary                      |
| Handles Outliers?         | No (assigns all points)        | Yes (marks as noise)           |
| Sensitive to Density?     | No                             | Yes                            |
| Time Complexity           | O(n * K * iterations)          | O(n log n) with indexing       |
| Best For                  | Compact, well-separated data   | Density-based, irregular data  |
+---------------------------+--------------------------------+--------------------------------+

FOR EARTHQUAKE DATA: DBSCAN is generally more suitable!
"""
print(summary_table)

# =============================================================================
# LIST OF ALL GENERATED FILES
# =============================================================================
print("\n" + "="*70)
print("ALL GENERATED FILES")
print("="*70)
print("""
Images:
  1. 01_earthquake_distribution.png    - Global earthquake map
  2. 02_feature_distributions.png      - Histograms of features
  3. 03_correlation_matrix.png         - Feature correlations
  4. 04_kmeans_elbow_silhouette.png    - Elbow method & silhouette
  5. 05_kmeans_clusters_map.png        - K-Means result map
  6. 06_dbscan_kdistance.png           - K-distance graph for eps
  7. 07_dbscan_clusters_map.png        - DBSCAN result map
  8. 08_comparison_side_by_side.png    - Side-by-side comparison
  9. 09_cluster_characteristics.png    - Cluster boxplots

Data:
  1. earthquake_clustering_results.csv - Full dataset with cluster labels
  2. algorithm_comparison.csv          - Comparison metrics table
""")

print("\n" + "="*70)
print("SCRIPT COMPLETED SUCCESSFULLY!")
print("="*70)
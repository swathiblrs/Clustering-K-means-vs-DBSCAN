"""
================================================================================
K-Means vs DBSCAN: E-Commerce Customer Behavior Dataset
================================================================================
COSC 527 - Biologically-Inspired Computation
Group Presentation - April 23, 2026

Dataset: E-commerce Customer Behavior Dataset (Kaggle)
Source: Paramita & Hariguna (2024) - Journal of Digital Market and Digital Currency
Features: 11 customer attributes
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

print("="*70)
print("K-Means vs DBSCAN: E-Commerce Customer Behavior Dataset")
print("="*70)

# =============================================================================
# 2. LOAD AND EXPLORE DATA
# =============================================================================
print("\n" + "="*70)
print("SECTION 2: Loading and Exploring Data")
print("="*70)

# Load dataset
df = pd.read_csv('E-commerce_Customer_Behavior_-_Sheet1.csv')

print(f"\nDataset Shape: {df.shape}")
print(f"Total Customers: {len(df)}")
print(f"\nFeatures Available ({len(df.columns)} columns):")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nData Types:")
print(df.dtypes)

print(f"\nStatistical Summary (Numerical):")
print(df.describe().round(2))

# Check for missing values
print(f"\nMissing Values:")
print(df.isnull().sum())

# =============================================================================
# 3. DATA PREPROCESSING
# =============================================================================
print("\n" + "="*70)
print("SECTION 3: Data Preprocessing")
print("="*70)

# Handle missing values
df = df.dropna()
print(f"After removing missing values: {len(df)} rows")

# Encode categorical variables
le_gender = LabelEncoder()
le_city = LabelEncoder()
le_membership = LabelEncoder()
le_discount = LabelEncoder()
le_satisfaction = LabelEncoder()

df['Gender_Encoded'] = le_gender.fit_transform(df['Gender'])
df['City_Encoded'] = le_city.fit_transform(df['City'])
df['Membership_Encoded'] = le_membership.fit_transform(df['Membership Type'])
df['Discount_Encoded'] = le_discount.fit_transform(df['Discount Applied'].astype(str))
df['Satisfaction_Encoded'] = le_satisfaction.fit_transform(df['Satisfaction Level'].fillna('Unknown'))

print(f"\nGender Encoding: {dict(zip(le_gender.classes_, range(len(le_gender.classes_))))}")
print(f"City Encoding: {dict(zip(le_city.classes_, range(len(le_city.classes_))))}")
print(f"Membership Encoding: {dict(zip(le_membership.classes_, range(len(le_membership.classes_))))}")

# Select features for clustering (numerical + encoded categorical)
feature_columns = ['Age', 'Total Spend', 'Items Purchased', 'Average Rating', 
                   'Days Since Last Purchase', 'Gender_Encoded', 'Membership_Encoded']

X = df[feature_columns].copy()

print(f"\nFeatures used for clustering: {len(feature_columns)}")
print(feature_columns)
print(f"Samples: {len(X)}")

# Normalize features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nAfter Scaling - Mean: ~0, Std: ~1")

# Apply PCA for visualization (reduce to 2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA for Visualization:")
print(f"  Original dimensions: {X_scaled.shape[1]}")
print(f"  Reduced dimensions: {X_pca.shape[1]}")
print(f"  Variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# =============================================================================
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
print("\n" + "="*70)
print("SECTION 4: Exploratory Data Analysis")
print("="*70)

# 4.1 Customer distribution by Total Spend vs Items Purchased
print("\nGenerating customer distribution plot...")
fig, ax = plt.subplots(figsize=(12, 8))

scatter = ax.scatter(
    df['Total Spend'], 
    df['Items Purchased'],
    c=df['Membership_Encoded'],
    cmap='viridis',
    s=100,
    alpha=0.7,
    edgecolors='black',
    linewidths=0.5
)

plt.colorbar(scatter, label='Membership Type (0=Bronze, 1=Gold, 2=Silver)')
ax.set_xlabel('Total Spend ($)', fontsize=12)
ax.set_ylabel('Items Purchased', fontsize=12)
ax.set_title('E-Commerce Customers: Total Spend vs Items Purchased\n(colored by Membership Type)', fontsize=14)

plt.tight_layout()
plt.savefig('01_customer_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 01_customer_distribution.png")

# 4.2 Distribution of key features
print("\nGenerating feature distribution plots...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
key_features = ['Age', 'Total Spend', 'Items Purchased', 'Average Rating', 
                'Days Since Last Purchase', 'Membership Type']

for i, (ax, feat) in enumerate(zip(axes.flatten(), key_features)):
    if feat == 'Membership Type':
        df[feat].value_counts().plot(kind='bar', ax=ax, color=plt.cm.viridis(0.5), edgecolor='black')
        ax.set_xlabel(feat)
        ax.set_ylabel('Count')
    else:
        ax.hist(df[feat], bins=20, color=plt.cm.viridis(i/6), edgecolor='black', alpha=0.7)
        ax.set_xlabel(feat)
        ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of {feat}')

plt.tight_layout()
plt.savefig('02_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 02_feature_distributions.png")

# 4.3 Correlation heatmap
print("\nGenerating correlation heatmap...")
plt.figure(figsize=(10, 8))
numerical_cols = ['Age', 'Total Spend', 'Items Purchased', 'Average Rating', 'Days Since Last Purchase']
correlation = df[numerical_cols].corr()
mask = np.triu(np.ones_like(correlation, dtype=bool))
sns.heatmap(correlation, mask=mask, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, fmt='.2f', annot_kws={'size': 10})
plt.title('Feature Correlation Matrix', fontsize=14)
plt.tight_layout()
plt.savefig('03_correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 03_correlation_matrix.png")

# 4.4 Pairplot
print("\nGenerating pairplot...")
pairplot_cols = ['Age', 'Total Spend', 'Items Purchased', 'Membership Type']
sns.pairplot(df[pairplot_cols + ['Gender']], hue='Gender', palette='husl', 
             vars=['Age', 'Total Spend', 'Items Purchased'])
plt.suptitle('Pairplot of Customer Features', y=1.02)
plt.savefig('04_pairplot.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 04_pairplot.png")

# =============================================================================
# 5. K-MEANS CLUSTERING
# =============================================================================
print("\n" + "="*70)
print("SECTION 5: K-Means Clustering")
print("="*70)

# 5.1 Finding Optimal K
print("\n5.1 Finding Optimal K...")
K_range = range(2, 11)
inertias = []
silhouette_scores_kmeans = []

print(f"\n{'K':<5} {'Inertia':<15} {'Silhouette':<15}")
print("-"*40)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    sil_score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores_kmeans.append(sil_score)
    print(f"{k:<5} {kmeans.inertia_:<15.2f} {sil_score:<15.4f}")

optimal_k = K_range[np.argmax(silhouette_scores_kmeans)]
print(f"\nOptimal K based on Silhouette: {optimal_k}")

# Plot Elbow and Silhouette
print("\nGenerating Elbow and Silhouette plots...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[0].set_ylabel('Inertia (Within-cluster Sum of Squares)', fontsize=12)
axes[0].set_title('Elbow Method for Optimal K', fontsize=14)
axes[0].axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal K={optimal_k}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(K_range, silhouette_scores_kmeans, 'go-', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[1].set_ylabel('Silhouette Score', fontsize=12)
axes[1].set_title('Silhouette Score vs K', fontsize=14)
axes[1].axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal K={optimal_k}')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('05_kmeans_elbow_silhouette.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 05_kmeans_elbow_silhouette.png")

# 5.2 Apply K-Means with Optimal K
K_OPTIMAL = optimal_k
print(f"\n5.2 Applying K-Means with K={K_OPTIMAL}...")

kmeans_final = KMeans(n_clusters=K_OPTIMAL, random_state=42, n_init=10)
kmeans_labels = kmeans_final.fit_predict(X_scaled)
df['kmeans_cluster'] = kmeans_labels

print(f"\nK-Means Cluster Distribution:")
print(df['kmeans_cluster'].value_counts().sort_index())

# K-Means Evaluation Metrics
kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
kmeans_calinski = calinski_harabasz_score(X_scaled, kmeans_labels)
kmeans_davies = davies_bouldin_score(X_scaled, kmeans_labels)

print(f"\nK-Means Metrics (K={K_OPTIMAL}):")
print(f"  Silhouette Score: {kmeans_silhouette:.4f}")
print(f"  Calinski-Harabasz Index: {kmeans_calinski:.2f}")
print(f"  Davies-Bouldin Index: {kmeans_davies:.4f}")

# Visualize K-Means in PCA space
print("\nGenerating K-Means cluster plot...")
fig, ax = plt.subplots(figsize=(12, 8))

colors = plt.cm.tab10(np.linspace(0, 1, K_OPTIMAL))
for i in range(K_OPTIMAL):
    mask = kmeans_labels == i
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[colors[i]], s=100, alpha=0.7, 
               edgecolors='black', linewidths=0.5, label=f'Cluster {i} (n={mask.sum()})')

# Plot cluster centers
centers_pca = pca.transform(kmeans_final.cluster_centers_)
ax.scatter(centers_pca[:, 0], centers_pca[:, 1], c='black', marker='X', s=300, 
           edgecolors='white', linewidths=2, label='Cluster Centers', zorder=5)

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
ax.set_title(f'K-Means Clustering (K={K_OPTIMAL}) in PCA Space', fontsize=14)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('06_kmeans_clusters.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 06_kmeans_clusters.png")

# =============================================================================
# 6. DBSCAN CLUSTERING
# =============================================================================
print("\n" + "="*70)
print("SECTION 6: DBSCAN Clustering")
print("="*70)

# 6.1 Finding Optimal eps using k-distance graph
min_samples = 5
print(f"\n6.1 Finding Optimal eps (min_samples={min_samples})...")

neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors_fit = neighbors.fit(X_scaled)
distances, _ = neighbors_fit.kneighbors(X_scaled)
k_distances = np.sort(distances[:, min_samples-1])

print("\nGenerating K-Distance graph...")
plt.figure(figsize=(10, 6))
plt.plot(range(len(k_distances)), k_distances, 'b-', linewidth=2)
plt.xlabel('Points (sorted by distance)', fontsize=12)
plt.ylabel(f'{min_samples}-th Nearest Neighbor Distance', fontsize=12)
plt.title('K-Distance Graph for eps Selection', fontsize=14)

# Add reference lines
for eps_val in [0.5, 1.0, 1.5, 2.0]:
    plt.axhline(y=eps_val, color='red', linestyle='--', alpha=0.5, label=f'eps={eps_val}')

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('07_dbscan_kdistance.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 07_dbscan_kdistance.png")

print(f"\nPercentile distances:")
print(f"  90th percentile: {np.percentile(k_distances, 90):.3f}")
print(f"  95th percentile: {np.percentile(k_distances, 95):.3f}")

# 6.2 Test Different eps Values
print("\n6.2 Testing eps values...")
print(f"\n{'eps':<8} {'Clusters':<10} {'Noise':<10} {'Noise %':<10} {'Silhouette':<12}")
print("-"*55)

best_eps = 1.0
best_sil = -1
dbscan_results = []

for eps in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]:
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    noise_pct = (n_noise / len(labels)) * 100
    
    mask = labels != -1
    if n_clusters > 1 and sum(mask) > n_clusters:
        sil = silhouette_score(X_scaled[mask], labels[mask])
        if sil > best_sil:
            best_sil = sil
            best_eps = eps
    else:
        sil = -1
    
    dbscan_results.append({
        'eps': eps, 'n_clusters': n_clusters, 'n_noise': n_noise, 
        'noise_pct': noise_pct, 'silhouette': sil
    })
    print(f"{eps:<8} {n_clusters:<10} {n_noise:<10} {noise_pct:<10.1f} {sil:<12.4f}")

# 6.3 Apply DBSCAN with Best eps
EPS_OPTIMAL = best_eps
MIN_SAMPLES = min_samples
print(f"\n6.3 Applying DBSCAN (eps={EPS_OPTIMAL}, min_samples={MIN_SAMPLES})...")

dbscan_final = DBSCAN(eps=EPS_OPTIMAL, min_samples=MIN_SAMPLES)
dbscan_labels = dbscan_final.fit_predict(X_scaled)
df['dbscan_cluster'] = dbscan_labels

n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

print(f"\nDBSCAN Results:")
print(f"  Clusters: {n_clusters_dbscan}")
print(f"  Noise points: {n_noise} ({(n_noise/len(dbscan_labels))*100:.1f}%)")
print(f"\nCluster Distribution:")
print(df['dbscan_cluster'].value_counts().sort_index())

# DBSCAN Evaluation Metrics
mask = dbscan_labels != -1
if sum(mask) > n_clusters_dbscan and n_clusters_dbscan > 1:
    dbscan_silhouette = silhouette_score(X_scaled[mask], dbscan_labels[mask])
    dbscan_calinski = calinski_harabasz_score(X_scaled[mask], dbscan_labels[mask])
    dbscan_davies = davies_bouldin_score(X_scaled[mask], dbscan_labels[mask])
    print(f"\nDBSCAN Metrics (excluding noise):")
    print(f"  Silhouette Score: {dbscan_silhouette:.4f}")
    print(f"  Calinski-Harabasz Index: {dbscan_calinski:.2f}")
    print(f"  Davies-Bouldin Index: {dbscan_davies:.4f}")
else:
    dbscan_silhouette = -1
    dbscan_calinski = 0
    dbscan_davies = float('inf')
    print("\nNot enough clusters to calculate metrics.")

# Visualize DBSCAN in PCA space
print("\nGenerating DBSCAN cluster plot...")
fig, ax = plt.subplots(figsize=(12, 8))

noise_mask = dbscan_labels == -1
unique_labels = sorted(set(dbscan_labels) - {-1})
colors_db = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 1) + 1))

for i, label in enumerate(unique_labels):
    cluster_mask = dbscan_labels == label
    ax.scatter(X_pca[cluster_mask, 0], X_pca[cluster_mask, 1], c=[colors_db[i]], s=100, 
               alpha=0.7, edgecolors='black', linewidths=0.5, label=f'Cluster {label} (n={cluster_mask.sum()})')

if n_noise > 0:
    ax.scatter(X_pca[noise_mask, 0], X_pca[noise_mask, 1], c='gray', s=50, alpha=0.5, 
               marker='x', label=f'Noise (n={n_noise})')

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
ax.set_title(f'DBSCAN Clustering (eps={EPS_OPTIMAL}, min_samples={MIN_SAMPLES})', fontsize=14)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('08_dbscan_clusters.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 08_dbscan_clusters.png")

# =============================================================================
# 7. SIDE-BY-SIDE COMPARISON
# =============================================================================
print("\n" + "="*70)
print("SECTION 7: Side-by-Side Comparison")
print("="*70)

print("\nGenerating side-by-side comparison plot...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# K-Means plot
for i in range(K_OPTIMAL):
    mask = kmeans_labels == i
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], c=[colors[i]], s=80, alpha=0.7, 
                    edgecolors='black', linewidths=0.5, label=f'Cluster {i}')
axes[0].scatter(centers_pca[:, 0], centers_pca[:, 1], c='black', marker='X', s=200, 
                edgecolors='white', linewidths=2, label='Centers', zorder=5)
axes[0].set_xlabel('PC1', fontsize=12)
axes[0].set_ylabel('PC2', fontsize=12)
axes[0].set_title(f'K-Means (K={K_OPTIMAL})\nSilhouette: {kmeans_silhouette:.4f}', fontsize=13)
axes[0].legend(loc='upper right', fontsize=9)
axes[0].grid(True, alpha=0.3)

# DBSCAN plot
for i, label in enumerate(unique_labels):
    cluster_mask = dbscan_labels == label
    axes[1].scatter(X_pca[cluster_mask, 0], X_pca[cluster_mask, 1], c=[colors_db[i]], s=80, 
                    alpha=0.7, edgecolors='black', linewidths=0.5, label=f'Cluster {label}')
if n_noise > 0:
    axes[1].scatter(X_pca[noise_mask, 0], X_pca[noise_mask, 1], c='gray', s=40, alpha=0.5, 
                    marker='x', label=f'Noise ({n_noise})')

sil_str = f'{dbscan_silhouette:.4f}' if dbscan_silhouette > -1 else 'N/A'
axes[1].set_xlabel('PC1', fontsize=12)
axes[1].set_ylabel('PC2', fontsize=12)
axes[1].set_title(f'DBSCAN (eps={EPS_OPTIMAL})\nSilhouette: {sil_str}', fontsize=13)
axes[1].legend(loc='upper right', fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('09_comparison_side_by_side.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 09_comparison_side_by_side.png")

# Comparison Table
print("\n" + "="*70)
print("COMPARISON TABLE: K-Means vs DBSCAN")
print("="*70)

print(f"""
{'Metric':<25} {'K-Means':<20} {'DBSCAN':<20}
{'-'*65}
{'Clusters':<25} {K_OPTIMAL:<20} {n_clusters_dbscan:<20}
{'Noise Points':<25} {'0':<20} {f'{n_noise} ({(n_noise/len(dbscan_labels))*100:.1f}%)':<20}
{'Silhouette Score':<25} {f'{kmeans_silhouette:.4f}':<20} {sil_str:<20}
{'Calinski-Harabasz':<25} {f'{kmeans_calinski:.2f}':<20} {f'{dbscan_calinski:.2f}' if dbscan_calinski > 0 else 'N/A':<20}
{'Davies-Bouldin':<25} {f'{kmeans_davies:.4f}':<20} {f'{dbscan_davies:.4f}' if dbscan_davies < float('inf') else 'N/A':<20}
{'Requires K?':<25} {'Yes':<20} {'No':<20}
{'Handles Outliers?':<25} {'No':<20} {'Yes':<20}
""")

# =============================================================================
# 8. CLUSTER ANALYSIS & INTERPRETATION
# =============================================================================
print("\n" + "="*70)
print("SECTION 8: Cluster Analysis & Customer Interpretation")
print("="*70)

# K-Means cluster statistics
print("\n8.1 K-Means Cluster Statistics:")
print("-"*60)
cluster_analysis = ['Total Spend', 'Items Purchased', 'Average Rating', 'Days Since Last Purchase', 'Age']
kmeans_stats = df.groupby('kmeans_cluster')[cluster_analysis].mean()
print(kmeans_stats.round(2))

# Cluster interpretation
print("\n8.2 K-Means Cluster Interpretation:")
print("-"*60)

for i in range(K_OPTIMAL):
    cluster_data = df[df['kmeans_cluster'] == i]
    avg_spend = cluster_data['Total Spend'].mean()
    avg_items = cluster_data['Items Purchased'].mean()
    avg_rating = cluster_data['Average Rating'].mean()
    avg_recency = cluster_data['Days Since Last Purchase'].mean()
    
    # Classify customer type
    if avg_spend > 1200 and avg_rating > 4.5:
        customer_type = "Premium Loyal Customers"
    elif avg_spend > 700 and avg_rating > 4.0:
        customer_type = "Regular Customers"
    elif avg_spend < 550 and avg_recency > 35:
        customer_type = "At-Risk Customers"
    else:
        customer_type = "Standard Customers"
    
    print(f"\nCluster {i}: {customer_type}")
    print(f"  - Count: {len(cluster_data)} customers")
    print(f"  - Avg Spend: ${avg_spend:.2f}")
    print(f"  - Avg Items: {avg_items:.1f}")
    print(f"  - Avg Rating: {avg_rating:.2f}")
    print(f"  - Avg Days Since Purchase: {avg_recency:.0f}")

# Membership distribution by cluster
print("\n8.3 Membership Type Distribution by K-Means Cluster:")
print("-"*60)
membership_dist = pd.crosstab(df['kmeans_cluster'], df['Membership Type'], normalize='index') * 100
print(membership_dist.round(1))

# 8.4 Cluster characteristics boxplots
print("\nGenerating cluster characteristics plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

df.boxplot(column='Total Spend', by='kmeans_cluster', ax=axes[0, 0])
axes[0, 0].set_title('K-Means: Total Spend by Cluster')
axes[0, 0].set_xlabel('Cluster')
axes[0, 0].set_ylabel('Total Spend ($)')

df.boxplot(column='Items Purchased', by='kmeans_cluster', ax=axes[0, 1])
axes[0, 1].set_title('K-Means: Items Purchased by Cluster')
axes[0, 1].set_xlabel('Cluster')
axes[0, 1].set_ylabel('Items Purchased')

dbscan_valid = df[df['dbscan_cluster'] != -1]
if len(dbscan_valid) > 0 and n_clusters_dbscan > 0:
    dbscan_valid.boxplot(column='Total Spend', by='dbscan_cluster', ax=axes[1, 0])
    axes[1, 0].set_title('DBSCAN: Total Spend by Cluster')
    axes[1, 0].set_xlabel('Cluster')
    axes[1, 0].set_ylabel('Total Spend ($)')

    dbscan_valid.boxplot(column='Items Purchased', by='dbscan_cluster', ax=axes[1, 1])
    axes[1, 1].set_title('DBSCAN: Items Purchased by Cluster')
    axes[1, 1].set_xlabel('Cluster')
    axes[1, 1].set_ylabel('Items Purchased')

plt.suptitle('')
plt.tight_layout()
plt.savefig('10_cluster_characteristics.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 10_cluster_characteristics.png")

# =============================================================================
# 9. SAVE RESULTS
# =============================================================================
print("\n" + "="*70)
print("SECTION 9: Save Results")
print("="*70)

df.to_csv('ecommerce_clustering_results.csv', index=False)
print("Saved: ecommerce_clustering_results.csv")

# Save comparison table
comparison_data = {
    'Metric': ['Clusters', 'Noise Points', 'Silhouette Score', 'Calinski-Harabasz', 
               'Davies-Bouldin', 'Requires K?', 'Handles Outliers?'],
    'K-Means': [K_OPTIMAL, 0, kmeans_silhouette, kmeans_calinski, kmeans_davies, 'Yes', 'No'],
    'DBSCAN': [n_clusters_dbscan, n_noise, dbscan_silhouette if dbscan_silhouette > -1 else 'N/A', 
               dbscan_calinski if dbscan_calinski > 0 else 'N/A', 
               dbscan_davies if dbscan_davies < float('inf') else 'N/A', 'No', 'Yes']
}
comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv('algorithm_comparison.csv', index=False)
print("Saved: algorithm_comparison.csv")

# =============================================================================
# 10. FINAL SUMMARY
# =============================================================================
print("\n" + "="*70)
print("FINAL SUMMARY FOR PRESENTATION")
print("="*70)

print(f"""
================================================================================
E-COMMERCE CUSTOMER BEHAVIOR CLUSTERING RESULTS
================================================================================

DATASET:
  - Source: E-commerce Customer Behavior Dataset (Kaggle)
  - Paper: Paramita & Hariguna (2024)
  - Customers: {len(df)}
  - Features: {len(feature_columns)} (Age, Total Spend, Items, Rating, Recency, etc.)

K-MEANS (K={K_OPTIMAL}):
  - Silhouette Score: {kmeans_silhouette:.4f}
  - Calinski-Harabasz: {kmeans_calinski:.2f}
  - Davies-Bouldin: {kmeans_davies:.4f}
  - All customers assigned to clusters

DBSCAN (eps={EPS_OPTIMAL}, min_samples={MIN_SAMPLES}):
  - Clusters: {n_clusters_dbscan}
  - Noise: {n_noise} customers ({(n_noise/len(dbscan_labels))*100:.1f}%)
  - Silhouette: {dbscan_silhouette:.4f if dbscan_silhouette > -1 else 'N/A'}

COMPARISON WITH PAPER (Paramita & Hariguna, 2024):
  Paper Results:
    - K-Means Silhouette: 0.546
    - DBSCAN Silhouette: 0.680
  Our Results:
    - K-Means Silhouette: {kmeans_silhouette:.4f}
    - DBSCAN Silhouette: {dbscan_silhouette:.4f if dbscan_silhouette > -1 else 'N/A'}

CONCLUSION:
  - Both algorithms provide useful customer segmentation
  - K-Means: Better for clear, balanced customer groups
  - DBSCAN: Better for identifying outlier/unusual customers

FILES GENERATED:
  - 10 PNG images (01-10)
  - ecommerce_clustering_results.csv
  - algorithm_comparison.csv
================================================================================
""")

print("\n" + "="*70)
print("SCRIPT COMPLETED SUCCESSFULLY!")
print("="*70)
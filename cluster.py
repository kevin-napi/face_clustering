import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import umap
 
# CONFIG
OUTPUT_DIR      = Path("outputs")
EMBEDDINGS_PATH = OUTPUT_DIR / "embeddings.npy"
METADATA_PATH   = OUTPUT_DIR / "metadata.csv"
 
# Dimensionality reduction
PCA_COMPONENTS = 50
 
# UMAP
UMAP_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
 
CLUSTER_METHOD = "kmeans"
 
# KMeans settings
KMEANS_K = 8
 

DBSCAN_EPS = 0.5
DBSCAN_MIN_SAMPLES = 5
# ──────────────────────────────────────────────────────────────────────────────
 
 
def reduce_pca(embeddings: np.ndarray, n_components: int) -> np.ndarray:
    print(f"PCA: {embeddings.shape[1]} → {n_components} dims ...")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(embeddings)
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(scaled)
    explained = pca.explained_variance_ratio_.sum()
    print(f"  Variance explained by {n_components} PCs: {explained:.1%}")
    return reduced
 
 
def reduce_umap(pca_embeddings: np.ndarray) -> np.ndarray:
    print(f"UMAP: {pca_embeddings.shape[1]} → 2 dims ...")
    reducer = umap.UMAP(
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=2,
        random_state=42,
        verbose=True
    )
    coords = reducer.fit_transform(pca_embeddings)
    print(f"  UMAP output shape: {coords.shape}")
    return coords
 
 
def cluster_kmeans(coords: np.ndarray, k: int) -> np.ndarray:
    print(f"KMeans clustering (k={k}) ...")
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(coords)
    score = silhouette_score(coords, labels)
    print(f"  Silhouette score: {score:.4f}  (closer to 1.0 = better separation)")
    return labels
 
 
def cluster_dbscan(coords: np.ndarray) -> np.ndarray:
    print(f"DBSCAN clustering (eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES}) ...")
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    labels = db.fit_predict(coords)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    print(f"  Clusters found: {n_clusters}  |  Noise points: {n_noise}")
    return labels
 
 
def find_best_k(coords: np.ndarray, k_range=range(4, 15)) -> None:
    """Optional helper: print silhouette scores to help pick K."""
    print("\nSilhouette scores for different K values:")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(coords)
        score = silhouette_score(coords, labels)
        print(f"  k={k:2d} → {score:.4f}")
 
 
def main():
    # Load data
    embeddings = np.load(EMBEDDINGS_PATH)
    metadata   = pd.read_csv(METADATA_PATH)
    print(f"Loaded embeddings: {embeddings.shape}")
    print(f"Loaded metadata:   {len(metadata)} rows")
 
    pca_embeddings = reduce_pca(embeddings, PCA_COMPONENTS)
 
    umap_coords = reduce_umap(pca_embeddings)
 

    if CLUSTER_METHOD == "kmeans":

        labels = cluster_kmeans(umap_coords, KMEANS_K)
    elif CLUSTER_METHOD == "dbscan":
        labels = cluster_dbscan(umap_coords)
    else:
        raise ValueError(f"Unknown CLUSTER_METHOD: {CLUSTER_METHOD}")
 
    #Save results
    np.save(OUTPUT_DIR / "umap_coords.npy", umap_coords)
    np.save(OUTPUT_DIR / "cluster_labels.npy", labels)
 
    results = metadata.copy()
    results["umap_x"]  = umap_coords[:, 0]
    results["umap_y"]  = umap_coords[:, 1]
    results["cluster"] = labels
    results.to_csv(OUTPUT_DIR / "results.csv", index=False)
 
    print(f"\nResults saved → {OUTPUT_DIR / 'results.csv'}")
    print(results["cluster"].value_counts().sort_index().to_string())
 
 
if __name__ == "__main__":
    main()
# ======= umap_clustering.py =======
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from umap import UMAP
import hdbscan

# ────── Load both CSVs ──────
df_params = pd.read_csv("parameters.csv")
df_summary = pd.read_csv("simulation_summary.csv")

# ────── Merge by number_of_simulation ──────
df_summary = df_summary.rename(columns={"number_of_simulation": "sim_id"})
df_summary["sim_id"] = df_summary["sim_id"].astype(int)
df_params["sim_id"] = df_params.index + 1  # Ensure alignment

df = pd.merge(df_params, df_summary, on="sim_id")

# ────── Extract behavioral outcomes ──────
outcome_cols = ["avg_bacteria", "avg_phage", "avg_energy", "survival_time"]
X = df[outcome_cols].replace([np.inf, -np.inf], np.nan).dropna()
clean_idx = X.index  # keep track of which rows are valid
X_scaled = StandardScaler().fit_transform(X)

# ────── UMAP projection ──────
embedding = UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(X_scaled)

# ────── Clustering 1: HDBSCAN ──────
hdb = hdbscan.HDBSCAN(min_cluster_size=5)
labels_hdb = hdb.fit_predict(embedding)

# ────── Clustering 2: GMM ──────
gmm = GaussianMixture(n_components=4, random_state=42)
labels_gmm = gmm.fit_predict(embedding)

# ────── Store results ──────
df_combined = df.loc[clean_idx].copy()
df_combined["umap_x"] = embedding[:, 0]
df_combined["umap_y"] = embedding[:, 1]
df_combined["cluster_hdb"] = labels_hdb
df_combined["cluster_gmm"] = labels_gmm

# ────── Save combined data ──────
df_combined.to_csv("combined_data.csv", index=False)
print("✓ Saved → combined_data.csv")

# ────── Plot: UMAP colored by avg_bacteria ──────
plt.figure(figsize=(8,6))
sns.scatterplot(x="umap_x", y="umap_y", hue="avg_bacteria", data=df_combined,
                palette="viridis", s=50)
plt.title("UMAP Projection Colored by avg_bacteria")
plt.tight_layout()
plt.savefig("umap_behavior_plot.png")
plt.show()

# ────── Plot: UMAP colored by HDBSCAN cluster ──────
plt.figure(figsize=(8,6))
sns.scatterplot(x="umap_x", y="umap_y", hue="cluster_hdb", data=df_combined,
                palette="tab10", s=50)
plt.title("UMAP Projection Colored by HDBSCAN Cluster")
plt.tight_layout()
plt.savefig("umap_cluster_hdbscan.png")
plt.show()

# ────── Plot: UMAP colored by GMM cluster ──────
plt.figure(figsize=(8,6))
sns.scatterplot(x="umap_x", y="umap_y", hue="cluster_gmm", data=df_combined,
                palette="tab10", s=50)
plt.title("UMAP Projection Colored by GMM Cluster")
plt.tight_layout()
plt.savefig("umap_cluster_gmm.png")
plt.show()

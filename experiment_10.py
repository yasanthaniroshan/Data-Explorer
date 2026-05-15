import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


SEED = 42


file_name = os.path.splitext(os.path.basename(__file__))[0]
DATA_DIR = os.path.join(os.getcwd(), 'processed_data_60min_nsr_5min_af')
csv_file_name = '200x20_extracted_rr_intervals.csv'
csv_path = os.path.join(DATA_DIR, csv_file_name)
EXPORT_DIR = os.path.join(os.getcwd(), 'exported_plots', file_name)
os.makedirs(EXPORT_DIR, exist_ok=True)

df = pd.read_csv(csv_path)
# patients = df['patient_id'].unique()

# def categorize_time_to_event(time_to_event):
#     if time_to_event <= 600:
#         return 0
#     elif time_to_event <= 1200 and time_to_event > 600:
#         return 1
#     elif time_to_event <= 1800 and time_to_event > 1200:
#         return 2
#     elif time_to_event <= 2400 and time_to_event > 1800:
#         return 3
#     elif time_to_event <= 3000 and time_to_event > 2400:
#         return 4
#     elif time_to_event <= 3600 and time_to_event > 3000:
#         return 5
#     else:
#         return 6

FEATURES = ['RMSSD','pNN50','SDNN','alpha_1','sample_entropy','approximate_entropy']


X = df[FEATURES].values
y = df['TimeToEvent'].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# df['time_to_event_category'] = df['TimeToEvent'].apply(categorize_time_to_event)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10,8))
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='viridis', alpha=0.5)
plt.colorbar(scatter)
plt.title("PCA Projection of Features")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig(os.path.join(EXPORT_DIR,'pca_clusters.png'))

tsne = TSNE(n_components=2, perplexity=30, random_state=SEED)
X_tsne = tsne.fit_transform(X_scaled)
plt.figure(figsize=(10,8))
scatter = plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y, cmap='viridis', alpha=0.5)
plt.colorbar(scatter)
plt.title("t-SNE Projection")
plt.savefig(os.path.join(EXPORT_DIR,'tsne_clusters.png'))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

file_name = os.path.splitext(os.path.basename(__file__))[0]

DATA_DIR = os.path.join('/home','intellisense01','EML-Labs','datasets','Data-Explorer','processed_data_60min_nsr_5min_af_corrected')
csv_file_name = '200x20_extracted_rr_intervals.csv'
csv_path = os.path.join(DATA_DIR, csv_file_name)
EXPORT_DIR = os.path.join(os.getcwd(), 'exported_plots', file_name)
os.makedirs(EXPORT_DIR, exist_ok=True)

df = pd.read_csv(csv_path)
patients = df['patient_id'].unique()


segments_per_episode = df.groupby(['patient_id', 'episode_id']).size()

plt.figure(figsize=(10, 6))
plt.hist(segments_per_episode.values, bins=30, edgecolor='black', color='skyblue')
plt.axvline(np.percentile(segments_per_episode.values, 5), color='green', linestyle='--', linewidth=2, label='5th Percentile')
plt.axvline(np.percentile(segments_per_episode.values, 95), color='red', linestyle='--', linewidth=2, label='95th Percentile')
plt.title('Distribution of Segments per Episode')
plt.xlabel('Number of Segments')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.savefig(os.path.join(EXPORT_DIR, 'segments_per_episode_distribution.png'), dpi=300, bbox_inches='tight')
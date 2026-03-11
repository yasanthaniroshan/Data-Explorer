import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns

file_name = os.path.splitext(os.path.basename(__file__))[0]

DATA_DIR = os.path.join('/home','intellisense01','EML-Labs','datasets','Data-Explorer','processed_data_60min_nsr_5min_af_corrected')
csv_file_name = '200x20_extracted_rr_intervals.csv'
csv_path = os.path.join(DATA_DIR, csv_file_name)
EXPORT_DIR = os.path.join(os.getcwd(), 'exported_plots', file_name)
os.makedirs(EXPORT_DIR, exist_ok=True)

df = pd.read_csv(csv_path)
patients = df['patient_id'].unique()

MIN_SEGMENTS = 150

FEATURES = ['RMSSD','pNN50','SDNN','alpha_1','sample_entropy','approximate_entropy']

RMSSD_values = []
pNN50_values = []
SDNN_values = []
alpha_1_values = []
sample_entropy_values = []
approximate_entropy_values = []

for patient in patients:
    episodes = df[df['patient_id'] == patient]['episode_id'].unique()
    for episode in episodes:
        segments = df[(df['patient_id'] == patient) & (df['episode_id'] == episode)]
        if len(segments) < MIN_SEGMENTS:
            continue
        episode_data = df[(df['patient_id'] == patient) & (df['episode_id'] == episode)][FEATURES]
        RMSSD_values.append(episode_data['RMSSD'].values[-MIN_SEGMENTS:])
        pNN50_values.append(episode_data['pNN50'].values[-MIN_SEGMENTS:])
        SDNN_values.append(episode_data['SDNN'].values[-MIN_SEGMENTS:])
        alpha_1_values.append(episode_data['alpha_1'].values[-MIN_SEGMENTS:])
        sample_entropy_values.append(episode_data['sample_entropy'].values[-MIN_SEGMENTS:])
        approximate_entropy_values.append(episode_data['approximate_entropy'].values[-MIN_SEGMENTS:])

# Plotting the distribution of each feature
for feature, values in zip(FEATURES, [RMSSD_values, pNN50_values, SDNN_values, alpha_1_values, sample_entropy_values, approximate_entropy_values]):
    plt.figure(figsize=(10, 6))
    sns.histplot(np.concatenate(values), kde=True, bins=30)
    plt.title(f'Distribution of {feature} (Last {MIN_SEGMENTS} Segments)')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.grid()
    plt.savefig(os.path.join(EXPORT_DIR, f'{feature}_distribution_last_{MIN_SEGMENTS}_segments.png'), dpi=300, bbox_inches='tight')

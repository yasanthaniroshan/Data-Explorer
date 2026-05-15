import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
import random

random.seed(42)
np.random.seed(43)

file_name = os.path.splitext(os.path.basename(__file__))[0]

DATA_DIR = os.path.join(os.getcwd(), 'processed_data_60min_nsr_5min_af')
csv_file_name = '200x20_extracted_rr_intervals.csv'
csv_path = os.path.join(DATA_DIR, csv_file_name)
EXPORT_DIR = os.path.join(os.getcwd(), 'exported_plots', file_name)
os.makedirs(EXPORT_DIR, exist_ok=True)

df = pd.read_csv(csv_path)
patients = df['patient_id'].unique()

patient = np.random.choice(patients, size=1, replace=False)[0]
episodes = df[df['patient_id'] == patient]['episode_id'].unique()
episode = np.random.choice(episodes, size=1, replace=False)[0]
segments = df[(df['patient_id'] == patient) & (df['episode_id'] == episode) & (df['TimeToEvent'] > 0)].sort_values(by='TimeToEvent')

time_to_event = segments['TimeToEvent'].values/60 # Convert to minutes
rmssd_values = segments['RMSSD'].values
pnn50_values = segments['pNN50'].values
sdnn_values = segments['SDNN'].values
alpha_1_values = segments['alpha_1'].values
sample_entropy_values = segments['sample_entropy'].values
approximate_entropy_values = segments['approximate_entropy'].values
segments['file_name'] = segments.apply(lambda row: f"{row['patient_id']}_{row['episode_id']}_{row['segment_id']}", axis=1)
file_names = segments['file_name'].values
# print(segments[['file_name']].values)

DATASET_PATH = "/home/intellisense01/EML-Labs/datasets/iridia-af-records-v1.0.1"
RECORD_DIR = os.path.join(DATASET_PATH, patient)

print(os.listdir(RECORD_DIR))  # List all files in the patient's directory
data = []
for file_name in file_names:
    rr_intervals = np.load(os.path.join(DATA_DIR, f"{file_name}.npy"))
    data.append(rr_intervals)

data = np.array(data)
# print("Data shape:", data.shape)

plt.figure(figsize=(12, 8))
plt.plot(data[0])  # Plot the first segment's RR intervals
plt.title(f'RR Intervals for Patient {patient}, Episode {episode}')
plt.xlabel('Sample Index')
plt.ylabel('RR Interval (ms)')
plt.grid()
plt.savefig(os.path.join(EXPORT_DIR, f'rr_intervals_patient_{patient}_episode_{episode}.png'), dpi=300, bbox_inches='tight')

# ticks = np.linspace(0, 60, 7)   # 0,10,20,...,60
# labels = ticks[::-1]            # 60,50,...,0


# plt.figure(figsize=(12, 8))
# plt.suptitle(f'Feature Trends Over Time for Patient {patient}, Episode {episode}')

# plt.subplot(2, 3, 1)
# sns.lineplot(x=time_to_event, y=rmssd_values, marker='o',sort=False)
# plt.title(f'RMSSD')
# plt.xlabel('Time to Event')
# plt.ylabel('RMSSD')
# plt.xticks(ticks, labels)

# plt.subplot(2, 3, 2)
# sns.lineplot(x=time_to_event, y=pnn50_values, marker='o',sort=False)
# plt.title(f'pNN50')
# plt.xlabel('Time to Event')
# plt.ylabel('pNN50')
# plt.xticks(ticks, labels)


# plt.subplot(2, 3, 3)
# sns.lineplot(x=time_to_event, y=sdnn_values, marker='o',sort=False)
# plt.title(f'SDNN')
# plt.xlabel('Time to Event')
# plt.ylabel('SDNN')
# plt.xticks(ticks, labels)

# plt.subplot(2, 3, 4)
# sns.lineplot(x=time_to_event, y=alpha_1_values, marker='o',sort=False)
# plt.title(f'alpha_1')
# plt.xlabel('Time to Event')
# plt.ylabel('alpha_1')
# plt.xticks(ticks, labels)   

# plt.subplot(2, 3, 5)
# sns.lineplot(x=time_to_event, y=sample_entropy_values, marker='o',sort=False)
# plt.title(f'Sample Entropy')   
# plt.xlabel('Time to Event')
# plt.ylabel('Sample Entropy')
# plt.xticks(ticks, labels)

# plt.subplot(2, 3, 6)
# sns.lineplot(x=time_to_event, y=approximate_entropy_values, marker='o',sort=False) 
# plt.title(f'Approximate Entropy')
# plt.xlabel('Time to Event')
# plt.ylabel('Approximate Entropy')
# plt.xticks(ticks, labels)

# plt.tight_layout()
# plt.savefig(os.path.join(EXPORT_DIR, f'features_over_time_patient_{patient}_episode_{episode}.png'), dpi=300, bbox_inches='tight')


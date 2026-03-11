import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
import random

SEED = random.randint(0, 200)

# random.seed(SEED)
np.random.seed(SEED)
TIME_THRESHOLD = 30  # minutes

file_name = os.path.splitext(os.path.basename(__file__))[0]

DATA_DIR = os.path.join('/home','intellisense01','EML-Labs','datasets','Data-Explorer','processed_data_60min_nsr_5min_af_corrected')
csv_file_name = '200x20_extracted_rr_intervals.csv'
csv_path = os.path.join(DATA_DIR, csv_file_name)
EXPORT_DIR = os.path.join(os.getcwd(), 'exported_plots', file_name)
os.makedirs(EXPORT_DIR, exist_ok=True)

df = pd.read_csv(csv_path)
patients = df['patient_id'].unique()

patient = np.random.choice(patients, size=1, replace=False)[0]
episodes = df[df['patient_id'] == patient]['episode_id'].unique()
episode = np.random.choice(episodes, size=1, replace=False)[0]
segments = df[(df['patient_id'] == patient) & (df['episode_id'] == episode) & (df['TimeToEvent'] <= TIME_THRESHOLD*60)].sort_values(by='TimeToEvent')

time_to_event = segments['TimeToEvent'].values/60 # Convert to minutes
rmssd_values = segments['RMSSD'].values
pnn50_values = segments['pNN50'].values
sdnn_values = segments['SDNN'].values
alpha_1_values = segments['alpha_1'].values
sample_entropy_values = segments['sample_entropy'].values
approximate_entropy_values = segments['approximate_entropy'].values

ticks = np.linspace(0, TIME_THRESHOLD, 7)   # 0,10,20,...,60
labels = ticks[::-1]            # 60,50,...,0


features = {
    "RMSSD": rmssd_values,
    "pNN50": pnn50_values,
    "SDNN": sdnn_values,
    "alpha_1": alpha_1_values,
    "Sample Entropy": sample_entropy_values,
    "Approximate Entropy": approximate_entropy_values
}

plt.figure(figsize=(12, 8))
plt.suptitle(f'Feature Trends Over Time for Patient {patient}, Episode {episode}')

for i, (name, values) in enumerate(features.items(), 1):

    plt.subplot(2, 3, i)

    # plot raw values
    sns.lineplot(x=time_to_event, y=values, marker='o', sort=False)

    # fit linear trend
    slope, intercept = np.polyfit(time_to_event, values, 1)
    trend = slope * time_to_event + intercept

    # plot trend line
    plt.plot(time_to_event, trend, color='red', linewidth=2,
             label=f"slope={slope:.4f}")

    plt.title(f'{name}')
    plt.xlabel('Time to Event')
    plt.ylabel(name)

    plt.xticks(ticks, labels)
    plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(EXPORT_DIR,
    f'features_over_time_patient_{patient}_episode_{episode}.png'),
    dpi=300,
    bbox_inches='tight'
)
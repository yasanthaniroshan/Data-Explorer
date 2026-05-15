import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns

file_name = os.path.splitext(os.path.basename(__file__))[0]

DATA_DIR = os.path.join(os.getcwd(), 'processed_data_60min_nsr_5min_af')
csv_file_name = '200x20_extracted_rr_intervals.csv'
csv_path = os.path.join(DATA_DIR, csv_file_name)
EXPORT_DIR = os.path.join(os.getcwd(), 'exported_plots', file_name)
os.makedirs(EXPORT_DIR, exist_ok=True)

df = pd.read_csv(csv_path)
patients = df['patient_id'].unique()

common_time = np.linspace(0, 60, 150)   # 150 time points from 0–60 minutes

FEATURES = ['RMSSD','pNN50','SDNN','alpha_1','sample_entropy','approximate_entropy']

time_to_event_values = []
RMSSD_values = []
pNN50_values = []
SDNN_values = []
alpha_1_values = []
sample_entropy_values = []
approximate_entropy_values = []


def align_series(time_values, feature_values, common_time):
    aligned = []

    for t, f in zip(time_values, feature_values):
        # sort just in case
        idx = np.argsort(t)
        t_sorted = t[idx]
        f_sorted = f[idx]

        interp = np.interp(common_time, t_sorted, f_sorted)
        aligned.append(interp)

    return np.array(aligned)

for patient in patients:
    episodes = df[df['patient_id'] == patient]['episode_id'].unique()
    for episode in episodes:
        segments = df[(df['patient_id'] == patient) & (df['episode_id'] == episode) & (df['TimeToEvent'] > 0)]
        episode_data = df[(df['patient_id'] == patient) & (df['episode_id'] == episode) & (df['TimeToEvent'] > 0)].sort_values(by='TimeToEvent')[FEATURES]
        RMSSD_values.append(episode_data['RMSSD'].values)
        pNN50_values.append(episode_data['pNN50'].values)
        SDNN_values.append(episode_data['SDNN'].values)
        alpha_1_values.append(episode_data['alpha_1'].values)
        sample_entropy_values.append(episode_data['sample_entropy'].values)
        approximate_entropy_values.append(episode_data['approximate_entropy'].values)
        time_to_event_values.append(segments['TimeToEvent'].values/60) # Convert to minutes


RMSSD_values = align_series(time_to_event_values, RMSSD_values, common_time)
pNN50_values = align_series(time_to_event_values, pNN50_values, common_time)
SDNN_values = align_series(time_to_event_values, SDNN_values, common_time)
alpha_1_values = align_series(time_to_event_values, alpha_1_values, common_time)
sample_entropy_values = align_series(time_to_event_values, sample_entropy_values, common_time)
approximate_entropy_values = align_series(time_to_event_values, approximate_entropy_values, common_time)

def mean_ci(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    n = data.shape[0]
    ci = 1.96 * std / np.sqrt(n)
    return mean, mean-ci, mean+ci

# Plotting the Mean 95% Confidence Interval for each feature
for feature, values in zip(FEATURES, [
    RMSSD_values, pNN50_values, SDNN_values,
    alpha_1_values, sample_entropy_values, approximate_entropy_values
]):

    mean, ci_lower, ci_upper = mean_ci(values)

    plt.figure(figsize=(10,6))

    plt.plot(common_time, mean, label='Mean', color='blue')
    plt.fill_between(common_time, ci_lower, ci_upper,
                     color='blue', alpha=0.2, label='95% CI')

    plt.title(f'{feature} vs Time to Event')
    plt.xlabel('Time to Event (minutes)')
    plt.ylabel(feature)
    # Create ticks
    ticks = np.linspace(0, 60, 7)   # 0,10,20,...,60
    labels = ticks[::-1]            # 60,50,...,0

    plt.xticks(ticks, labels)
    plt.grid(True)
    plt.legend()

    plt.savefig(os.path.join(
        EXPORT_DIR, f'{feature}_mean_ci_time_to_event.png'
    ), dpi=300, bbox_inches='tight')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

file_name = os.path.splitext(os.path.basename(__file__))[0]


DATA_DIR = os.path.join('/home','intellisense01','EML-Labs','datasets','Data-Explorer','processed_data_60min_nsr_5min_af_corrected')
csv_file_name = '200x20_extracted_rr_intervals.csv'
csv_path = os.path.join(DATA_DIR, csv_file_name)
EXPORT_DIR = os.path.join(os.getcwd(), 'exported_plots', file_name)
os.makedirs(EXPORT_DIR, exist_ok=True)

df = pd.read_csv(csv_path)
patients = df['patient_id'].unique()

episodes_per_patient = df.groupby('patient_id')['episode_id'].nunique()

plt.figure(figsize=(20, 6))
sns.barplot(x=episodes_per_patient.index, y=episodes_per_patient.values, hue=episodes_per_patient.index, palette='viridis', legend=False)
plt.title('Number of Episodes per Patient')
plt.xlabel('Patient ID')
plt.ylabel('Number of Episodes')
plt.xticks(rotation=45)
plt.grid()
plt.savefig(os.path.join(EXPORT_DIR, 'episodes_per_patient.png'), dpi=300, bbox_inches='tight')

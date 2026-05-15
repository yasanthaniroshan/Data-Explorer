import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
import random
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


SEED = 42


file_name = os.path.splitext(os.path.basename(__file__))[0]
DATA_DIR = os.path.join(os.getcwd(), 'processed_data_60min_nsr_5min_af')
csv_file_name = '200x20_extracted_rr_intervals.csv'
csv_path = os.path.join(DATA_DIR, csv_file_name)
EXPORT_DIR = os.path.join(os.getcwd(), 'exported_plots', file_name)
os.makedirs(EXPORT_DIR, exist_ok=True)

df = pd.read_csv(csv_path)
patients = df['patient_id'].unique()

max_time_to_event = df['TimeToEvent'].max()
print("Max TimeToEvent:", max_time_to_event)
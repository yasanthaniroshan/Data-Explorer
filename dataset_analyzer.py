from typing import NoReturn
import neurokit2 as nk
import numpy as np
import pandas as pd
import os
from logging import getLogger
import logging
from tqdm import tqdm
import h5py
import csv

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('dataset_analyzer.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

DATASET_PATH = "/home/intellisense01/EML-Labs/datasets/iridia-af-records-v1.0.1"
PATIENT_LIST = sorted(os.listdir(DATASET_PATH))
AFIB_LENGTH = 60 * 60
SR_LENGTH = 60 * 60
PADDING = 60 * 60
WINDOW_SIZE = 200
STRIDE = 200
# Peaks from nk.intervals_to_peaks(..., sampling_rate=...) are in samples, not ms.
SAMPLING_RATE = 200

def load_episode_details(patient_list: list, dataset_path: str, afib_length: int, sr_length: int, padding: int)->dict:
    """
    Load all eligible AF episodes for each patient.

    Returns dict {patient_id: [list of episode dicts]} where episodes are
    ordered by row index (assumed chronological in the CSV).

    Args:
        patient_list: The list of patients to load.
        dataset_path: The path to the dataset.
        afib_length: The length of the AF episode in seconds.
        sr_length: The length of the SR episode in seconds.
        padding: The padding to add to the SR episode in seconds.

    Returns:
        A dictionary of patient episodes.
    """
    patient_episodes = {}
    for patient in patient_list:
        record_dir = os.path.join(dataset_path, patient)
        ecg_csv_file = f"{patient}_ecg_labels.csv"
        ecg_df = pd.read_csv(os.path.join(record_dir, ecg_csv_file))
        episodes = []
        for idx, row in ecg_df.iterrows():
            if (row['af_duration'] >= afib_length) and \
               (row['nsr_before_duration'] >= sr_length + padding):
                episodes.append({
                    "patient": patient,
                    "record_index": idx,
                    "start_file_index": row['start_file_index'],
                    "end_file_index": row['end_file_index'],
                })
        if episodes:
            patient_episodes[patient] = episodes
    return patient_episodes

def _load_rr_and_nsr(patient: str, episode: dict, dataset_path: str)->np.ndarray:
    """
    Load the full RR array and return NSR portion before AF onset (chronological).

    Args:
        patient: The patient ID.
        episode: The episode dictionary.
        dataset_path: The path to the dataset.

    Returns:
        The NSR data.
    """
    record_dir = os.path.join(dataset_path, patient)
    rr_df = pd.read_csv(os.path.join(record_dir, f"{patient}_rr_labels.csv"))
    row = rr_df.loc[episode['record_index']]
    
    rr_start_index = int(row['start_rr_index'])
    file_index = int(row['start_file_index'])

    with h5py.File(os.path.join(record_dir, f"{patient}_rr_{file_index:02d}.h5"), 'r') as f:
        rr_data = f['rr'][:]

    nsr_data = rr_data[:rr_start_index]
    return np.array(nsr_data)

def get_hrv_matrics(r_peaks: np.ndarray,sampling_rate:int=200) -> dict:
    data = nk.hrv_time(r_peaks, sampling_rate=sampling_rate).iloc[0].to_dict()
    return data
    
def get_freq_metrics(r_peaks: np.ndarray,sampling_rate:int=200) -> dict:
    data = nk.hrv_frequency(r_peaks, sampling_rate=sampling_rate).iloc[0].to_dict()
    return data

def get_non_linear_metrics(r_peaks: np.ndarray,sampling_rate:int=200) -> dict:
    data = nk.hrv_nonlinear(r_peaks, sampling_rate=sampling_rate).iloc[0].to_dict()
    return data

def get_rqa_metrics(r_peaks: np.ndarray,sampling_rate:int=200) -> dict:
    data = nk.hrv_rqa(r_peaks, sampling_rate=sampling_rate).iloc[0].to_dict()
    return data

def filter_episodes(rr_intervals:np.ndarray,sr_length:int)->np.ndarray:
    """
    Filter episodes based on the length of the NSR portion before AF onset.
    """
    nsr_reversed = rr_intervals[::-1]
    sr_cum_sum = np.cumsum(nsr_reversed)/1000 # convert to seconds
    sr_boundary_index = int(np.searchsorted(sr_cum_sum, sr_length))
    return np.array(rr_intervals[:sr_boundary_index])[::-1]

def segment_data(data:np.ndarray,window_size:int,stride:int)->np.ndarray:
    """
    Segment data into windows.
    """
    segments = []
    for i in range(0, len(data) - window_size + 1, stride):
        segments.append(data[i:i + window_size])
    return segments

def get_metrics(data:np.ndarray)->dict:
    """
    Get the metrics for the data.
    """
    hrv_metrics = get_hrv_matrics(data)
    freq_metrics = get_freq_metrics(data)
    non_linear_metrics = get_non_linear_metrics(data)
    rqa_metrics = get_rqa_metrics(data)
    return {**hrv_metrics, **freq_metrics, **non_linear_metrics, **rqa_metrics}



def main():
    patient_episodes = load_episode_details(PATIENT_LIST, DATASET_PATH, AFIB_LENGTH, SR_LENGTH, PADDING)
    logger.info(f"Loaded {len(patient_episodes)} patient episodes")
    total_episodes = sum(len(episodes) for episodes in patient_episodes.values())
    pbar = tqdm(total=total_episodes, desc="Processing episodes", unit="episode")
    df = pd.DataFrame()
    HEADER = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_SDANN1', 'HRV_SDNNI1', 'HRV_SDANN2', 'HRV_SDNNI2', 'HRV_SDANN5', 'HRV_SDNNI5', 'HRV_RMSSD', 'HRV_SDSD', 'HRV_CVNN', 'HRV_CVSD', 'HRV_MedianNN', 'HRV_MadNN', 'HRV_MCVNN', 'HRV_IQRNN', 'HRV_SDRMSSD', 'HRV_Prc20NN', 'HRV_Prc80NN', 'HRV_pNN50', 'HRV_pNN20', 'HRV_MinNN', 'HRV_MaxNN', 'HRV_HTI', 'HRV_TINN', 'HRV_ULF', 'HRV_VLF', 'HRV_LF', 'HRV_HF', 'HRV_VHF', 'HRV_TP', 'HRV_LFHF', 'HRV_LFn', 'HRV_HFn', 'HRV_LnHF', 'HRV_SD1', 'HRV_SD2', 'HRV_SD1SD2', 'HRV_S', 'HRV_CSI', 'HRV_CVI', 'HRV_CSI_Modified', 'HRV_PIP', 'HRV_IALS', 'HRV_PSS', 'HRV_PAS', 'HRV_GI', 'HRV_SI', 'HRV_AI', 'HRV_PI', 'HRV_C1d', 'HRV_C1a', 'HRV_SD1d', 'HRV_SD1a', 'HRV_C2d', 'HRV_C2a', 'HRV_SD2d', 'HRV_SD2a', 'HRV_Cd', 'HRV_Ca', 'HRV_SDNNd', 'HRV_SDNNa', 'HRV_DFA_alpha1', 'HRV_MFDFA_alpha1_Width', 'HRV_MFDFA_alpha1_Peak', 'HRV_MFDFA_alpha1_Mean', 'HRV_MFDFA_alpha1_Max', 'HRV_MFDFA_alpha1_Delta', 'HRV_MFDFA_alpha1_Asymmetry', 'HRV_MFDFA_alpha1_Fluctuation', 'HRV_MFDFA_alpha1_Increment', 'HRV_DFA_alpha2', 'HRV_MFDFA_alpha2_Width', 'HRV_MFDFA_alpha2_Peak', 'HRV_MFDFA_alpha2_Mean', 'HRV_MFDFA_alpha2_Max', 'HRV_MFDFA_alpha2_Delta', 'HRV_MFDFA_alpha2_Asymmetry', 'HRV_MFDFA_alpha2_Fluctuation', 'HRV_MFDFA_alpha2_Increment', 'HRV_ApEn', 'HRV_SampEn', 'HRV_ShanEn', 'HRV_FuzzyEn', 'HRV_MSEn', 'HRV_CMSEn', 'HRV_RCMSEn', 'HRV_CD', 'HRV_HFD', 'HRV_KFD', 'HRV_LZC', 'HRV_Symbolic_EqualProb4_0V', 'HRV_Symbolic_EqualProb4_1V', 'HRV_Symbolic_EqualProb4_2LV', 'HRV_Symbolic_EqualProb4_2UV', 'RecurrenceRate', 'DiagRec', 'Determinism', 'DeteRec', 'L', 'Divergence', 'LEn', 'Laminarity', 'TrappingTime', 'VMax', 'VEn', 'W', 'WMax', 'WEn', 'patient', 'episode_index', 'segment_index', 'time_remaining']
    with open("dataset_matrics.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
    for patient, episodes in patient_episodes.items():
        for idx, episode in enumerate(episodes):
            nsr_data = _load_rr_and_nsr(patient, episode, DATASET_PATH)
            nsr_data = filter_episodes(nsr_data, SR_LENGTH)
            total_time = np.sum(nsr_data)/1000 # convert to seconds
            r_peaks = nk.intervals_to_peaks(nsr_data, sampling_rate=SAMPLING_RATE)
            nsr_data = segment_data(r_peaks, WINDOW_SIZE, STRIDE)
            for i, segment in enumerate(nsr_data):
                time_end_sec = segment[-1] / SAMPLING_RATE
                time_remaining = total_time - time_end_sec
                metrics = get_metrics(segment) # get metrics for the segment
                metrics['patient'] = patient
                metrics['episode_index'] = idx
                metrics['segment_index'] = i
                metrics['time_remaining'] = time_remaining/60 # convert to minutes
                df = pd.concat([df, pd.DataFrame([metrics])], ignore_index=True)
            df.to_csv("dataset_matrics.csv", mode='a',header=False, index=False)
            pbar.update(1)
    pbar.close()

# ecg = nk.ecg_simulate(duration=10*60, sampling_rate=200)
# signals, info = nk.ecg_peaks(ecg, sampling_rate=200)
# r_peaks = info['ECG_R_Peaks']
# window_size = 200
# segments = []
# stride = 50
# data = []
# for i in range(0, len(r_peaks) - window_size + 1, stride):
#     segments.append(r_peaks[i:i + window_size])
#     hrv_metrics = get_hrv_matrics(r_peaks[i:i + window_size])
#     freq_metrics = get_freq_metrics(r_peaks[i:i + window_size])
#     non_linear_metrics = get_non_linear_metrics(r_peaks[i:i + window_size])
#     rqa_metrics = get_rqa_metrics(r_peaks[i:i + window_size])
#     all_metrics = {**hrv_metrics, **freq_metrics, **non_linear_metrics, **rqa_metrics}
#     data.append(all_metrics)

# df = pd.DataFrame(data)
# print(df.describe())

if __name__ == "__main__":
    main()
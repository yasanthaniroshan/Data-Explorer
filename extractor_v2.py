import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
import random
import h5py
import neurokit2 as nk
from scipy.interpolate import CubicSpline


file_name = os.path.splitext(os.path.basename(__file__))[0]
DATA_DIR = os.path.join(os.getcwd(), 'processed_data_60min_nsr_5min_af')
csv_file_name = '200x20_extracted_rr_intervals.csv'
csv_path = os.path.join(DATA_DIR, csv_file_name)
EXPORT_DIR = os.path.join(os.getcwd(), 'exported_plots', file_name)
DATASET_PATH = "/home/intellisense01/EML-Labs/datasets/iridia-af-records-v1.0.1"
SAMPLING_RATE = 200
UPSAMPLE_RATE = 1000
REFINE_WINDOW = 50 # 50ms window for refinement at 1000Hz = ±50 samples
VALIDATE_WINDOW = 100 # 100ms window for validation at 1000Hz = ±100 samples
SEGMENT_SEC = 30
OVERLAP_SEC = 10
EDGE_DISCARD_SEC = 5

def get_csv_files(record_path:str):
    files = os.listdir(record_path)
    return [f for f in files if f.endswith('.csv') and "ecg_labels" in f]

def filter_indexes_from_csv(record_name:str,csv_path:str,minimum_af_duration:int,minimum_pre_af_duration:int):
    df = pd.read_csv(csv_path)
    indexes = []
    for index, row in df.iterrows():
        af_duration = row['af_duration']
        nsr_before_duration = row['nsr_before_duration']
        if af_duration >= minimum_af_duration and nsr_before_duration >= minimum_pre_af_duration:
            start_file_index = row['start_file_index']
            start_qrs_index = row['start_qrs_index']
            end_file_index = row['end_file_index']
            end_qrs_index = row['end_qrs_index']
            indexes.append((record_name,start_file_index, end_file_index,start_qrs_index, end_qrs_index))
    return indexes

def load_ecg_data(record_name:str,start_file_index:int,end_file_index:int,start_qrs_index:int,end_qrs_index:int):
    ecg_data = None
    if start_file_index == end_file_index:
        with h5py.File(os.path.join(DATASET_PATH,record_name, f"{record_name}_ecg_{start_file_index:02d}.h5"), 'r') as h5_file:
            ecg_data = h5_file['ecg'][start_qrs_index:end_qrs_index][:,0]
    return ecg_data

def get_initial_peaks(ecg_signal, sampling_rate):
    cleaned_ecg = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
    signals, info = nk.ecg_peaks(cleaned_ecg, sampling_rate=sampling_rate)
    rpeak_indices = info['ECG_R_Peaks']
    return rpeak_indices

def get_interpolated_ecg(ecg_signal, original_sampling_rate, target_sampling_rate):
    old_length = len(ecg_signal)
    new_length = int(old_length * target_sampling_rate / original_sampling_rate)
    x_original = np.arange(old_length)
    x_new = np.linspace(0, old_length - 1, new_length)
    cs = CubicSpline(x_original, ecg_signal)
    interpolated_ecg = cs(x_new)
    return interpolated_ecg

def get_peaks(interpolated_ecg,refine_window:int=REFINE_WINDOW,validate_window:int=VALIDATE_WINDOW):
    r_peaks_primary = nk.ecg_findpeaks(interpolated_ecg, sampling_rate=1000, method='khamis2016')['ECG_R_Peaks']
    r_peaks_secondary = nk.ecg_findpeaks(interpolated_ecg, sampling_rate=1000, method='neurokit')['ECG_R_Peaks']
    validated_peaks = []
    inaccurate_peaks = []
    for peak in r_peaks_primary:
        # Refinement Phase: find absolute local maximum within ±50ms window
        win_start = max(0, peak - refine_window)
        win_end = min(len(interpolated_ecg) - 1, peak + REFINE_WINDOW)
        refined_peak = win_start + np.argmax(interpolated_ecg[win_start:win_end + 1])
        # Validation Phase: check if secondary detector has a peak within ±100ms
        is_validated = np.any(np.abs(r_peaks_secondary - refined_peak) <= validate_window)
        if is_validated:
            validated_peaks.append(refined_peak)
        else:
            inaccurate_peaks.append(refined_peak)
    r_peaks = {'ECG_R_Peaks': np.array(validated_peaks)}
    print(f"Primary detections:    {len(r_peaks_primary)}")
    print(f"Validated peaks:       {len(validated_peaks)}")
    print(f"Flagged/excluded peaks:{len(inaccurate_peaks)}")
    return r_peaks

def correct_peaks(r_peaks, sampling_rate):
    _, rpeaks_corrected = nk.signal_fixpeaks(r_peaks['ECG_R_Peaks'], sampling_rate=sampling_rate, method='Kubios',iterative=True)
    return rpeaks_corrected

def detect_peaks_segment(segment):

    r_primary = nk.ecg_findpeaks(
        segment, sampling_rate=UPSAMPLE_RATE, method="khamis2016"
    )["ECG_R_Peaks"]

    r_secondary = nk.ecg_findpeaks(
        segment, sampling_rate=UPSAMPLE_RATE, method="neurokit"
    )["ECG_R_Peaks"]

    validated = []

    for peak in r_primary:

        win_start = max(0, peak - REFINE_WINDOW)
        win_end = min(len(segment) - 1, peak + REFINE_WINDOW)

        refined = win_start + np.argmax(segment[win_start : win_end + 1])

        is_valid = np.any(np.abs(r_secondary - refined) <= VALIDATE_WINDOW)

        if is_valid:
            validated.append(refined)

    return np.array(validated)

def remove_duplicate_peaks(peaks, min_distance=250):

    cleaned = [peaks[0]]

    for p in peaks[1:]:
        if p - cleaned[-1] > min_distance:
            cleaned.append(p)

    return np.array(cleaned)
def detect_peaks_segmentwise(ecg):

    segment_len = SEGMENT_SEC * UPSAMPLE_RATE
    overlap = OVERLAP_SEC * UPSAMPLE_RATE
    discard = EDGE_DISCARD_SEC * UPSAMPLE_RATE

    step = segment_len - overlap

    all_peaks = []

    for start in range(0, len(ecg), step):

        end = start + segment_len
        if end > len(ecg):
            end = len(ecg)

        segment = ecg[start:end]
        # signals,info = nk.ecg_peaks(segment, sampling_rate=UPSAMPLE_RATE)
        peaks = detect_peaks_segment(segment)
        # peaks = info['ECG_R_Peaks']

        # convert to global index
        peaks = peaks + start

        # discard boundary peaks
        valid_start = start + discard
        valid_end = end - discard

        peaks = peaks[(peaks >= valid_start) & (peaks <= valid_end)]

        all_peaks.extend(peaks)

        if end == len(ecg):
            break

    all_peaks = np.unique(all_peaks)
    all_peaks = np.sort(all_peaks)
    all_peaks = remove_duplicate_peaks(all_peaks, min_distance=int(0.25 * UPSAMPLE_RATE))
    return all_peaks


records = os.listdir(DATASET_PATH)

csv_files = [get_csv_files(os.path.join(DATASET_PATH, record)) for record in records]

print(f"Found {len(csv_files)} CSV files in dataset")

indexes = []

for record in records:

    record_path = os.path.join(DATASET_PATH, record)

    for csv_file in csv_files[records.index(record)]:

        csv_path = os.path.join(record_path, csv_file)

        file_indexes = filter_indexes_from_csv(
            record,
            csv_path,
            minimum_af_duration=int(60 * 60*2),
            minimum_pre_af_duration=int(60 * 60 * 2),
        )

        indexes.extend(file_indexes)


print(f"Extracted {len(indexes)} valid indexes")

index = indexes[2]
record_name,start_file_index, end_file_index,start_qrs_index, end_qrs_index = index
start_qrs_index_ = max(0, start_qrs_index - int(2*60*60*SAMPLING_RATE))
end_qrs_index = start_qrs_index   + int(2*60*60*SAMPLING_RATE)

print(index)

ecg_data = load_ecg_data(record_name,start_file_index, end_file_index,start_qrs_index_, end_qrs_index)
print(f"Loaded ECG data with {len(ecg_data)} samples at {SAMPLING_RATE} Hz")
interpolated_ecg = get_interpolated_ecg(ecg_data, SAMPLING_RATE, UPSAMPLE_RATE)

peaks = detect_peaks_segmentwise(interpolated_ecg)
# _, peaks = nk.signal_fixpeaks(
#     peaks,
#     sampling_rate=UPSAMPLE_RATE,
#     method="neurokit"
# )

print("Total peaks:", len(peaks))




os.makedirs(EXPORT_DIR, exist_ok=True)


plt.figure(figsize=(12, 6))
plt.plot(interpolated_ecg)
plt.scatter(peaks, interpolated_ecg[peaks], color="red", s=10)
plt.savefig(os.path.join(EXPORT_DIR, f"{index[0]}_ecg_peaks_segmented.png"))
plt.close()


rr_intervals = np.diff(peaks) / UPSAMPLE_RATE

np.save(os.path.join(EXPORT_DIR, f"{index[0]}_rr_intervals_segmented.npy"), rr_intervals)
np.save(os.path.join(EXPORT_DIR, f"{index[0]}_peaks_segmented.npy"), peaks)

print(f"Min RR interval: {np.min(rr_intervals):.3f} seconds")
print(f"Max RR interval: {np.max(rr_intervals):.3f} seconds")
print(f"Mean RR interval: {np.mean(rr_intervals):.3f} seconds")
print(f"Median RR interval: {np.median(rr_intervals):.3f} seconds")
print(f"Std RR interval: {np.std(rr_intervals):.3f} seconds")
print(f"RR count: {len(rr_intervals)}")

idx = np.where(rr_intervals > 3)[0]

for i in idx:

    rr = rr_intervals[i]

    peak1 = peaks[i]
    peak2 = peaks[i + 1]

    center = (peak1 + peak2) // 2
    window = 4000  # 4 seconds at 1000 Hz

    detected_peaks = peaks[(peaks >= center - window) & (peaks <= center + window)]
    # detected_peaks = detected_peaks - (center - window)  # Adjust to segment-relative indices
    start = max(0, center - window)
    end = min(len(interpolated_ecg), center + window)

    segment = interpolated_ecg[start:end]
    signals,info = nk.ecg_peaks(segment, sampling_rate=UPSAMPLE_RATE)
    quality = nk.ecg_quality(segment,rpeaks=info['ECG_R_Peaks'], sampling_rate=UPSAMPLE_RATE)
    print(f"Segment {i} quality: {np.mean(quality):.3f}")
    peaks_in_segment = info['ECG_R_Peaks'] + start
    plt.figure(figsize=(12,6))
    plt.plot(segment)

    # plot peaks relative to segment
    # plt.scatter(peak1 - start, interpolated_ecg[peak1], color="red", label="R peak")
    # plt.scatter(peak2 - start, interpolated_ecg[peak2], color="orange", label="Next R peak")
    plt.scatter(detected_peaks-start, interpolated_ecg[detected_peaks], color="blue", s=50, label="Already Detected peaks",alpha=0.5,marker='o')
    plt.scatter(peaks_in_segment - start, interpolated_ecg[peaks_in_segment], color="green", s=50, label="Newly Detected peaks",alpha=0.5,marker='x')
    plt.title(f"RR interval: {rr:.3f} seconds (Index {i})")
    plt.legend()

    plt.savefig(os.path.join(EXPORT_DIR, f"{index[0]}_long_rr_{i}.png"))
    plt.close()


plt.figure(figsize=(12, 6))
plt.plot(rr_intervals)
plt.savefig(os.path.join(EXPORT_DIR, f"{index[0]}_rr_intervals_segmented.png"))
plt.close()
import os
import csv
import pandas as pd
import numpy as np
import h5py
from pathlib import Path
import neurokit2 as nk
from tqdm import tqdm

DATASET_PATH = "/home/intellisense01/EML-Labs/datasets/iridia-af-records-v1.0.1"
RECORDS = os.listdir(DATASET_PATH)
SAMPLING_RATE = 200
ECG_LABEL_PATTERN = "ecg_labels"
OUTPUT_PATH = "/home/intellisense01/EML-Labs/datasets/Data-Explorer/processed_data_60min_nsr_5min_af"

# Thresholds
MIN_AF_DURATION = 5 * 60  # 5 minutes in seconds
MIN_NSR_DURATION = 60 * 60   # 60 minutes in seconds

def find_ecg_labels_csv(record_dir):
    """Find the ECG labels CSV file in the record directory."""
    files = os.listdir(record_dir)
    csv_files = [f for f in files if f.endswith('.csv')]
    
    # Find the file matching ECG_LABEL_PATTERN
    for csv_file in csv_files:
        if ECG_LABEL_PATTERN.lower() in csv_file.lower():
            return os.path.join(record_dir, csv_file)
    
    return None

def find_h5_files(record_dir):
    """Find all H5 files in the record directory (ECG or RR files)."""
    files = os.listdir(record_dir)
    h5_files = sorted([f for f in files if f.endswith('.h5')])
    return h5_files

def find_rr_h5_files(record_dir, record_name):
    """Find RR H5 files for a specific record."""
    files = os.listdir(record_dir)
    rr_files = sorted([f for f in files if f'rr' in f.lower() and f.endswith('.h5')])
    return rr_files

def find_ecg_h5_files(record_dir, record_name):
    """Find ECG H5 files for a specific record."""
    files = os.listdir(record_dir)
    ecg_files = sorted([f for f in files if 'ecg' in f.lower() and f.endswith('.h5')])
    return ecg_files

def load_rr_data_from_h5(h5_path):
    """Load RR interval data from H5 file."""
    try:
        with h5py.File(h5_path, 'r') as f:
            # List available datasets
            if 'rr' in f:
                rr_data = f['rr'][()]
            else:
                # Try to find the RR data with different key names
                keys = list(f.keys())
                if keys:
                    rr_data = f[keys[0]][()]
                else:
                    return None
        return rr_data
    except Exception as e:
        print(f"Error loading {h5_path}: {e}")
        return None

def load_h5_data(h5_path):
    """Load ECG data from H5 file."""
    try:
        with h5py.File(h5_path, 'r') as f:
            # List available datasets
            if 'ecg' in f:
                ecg_data = f['ecg'][:][:,1]
            else:
                # Try to find the ECG data with different key names
                keys = list(f.keys())
                if keys:
                    ecg_data = f[keys[0]][()][:,1]
                else:
                    return None
        return ecg_data
    except Exception as e:
        print(f"Error loading {h5_path}: {e}")
        return None

def find_RR_intervals(ecg_data):
    """Detect R-peaks and Calculate RR intervals in s"""
    try:
        _,info = nk.ecg_process(ecg_data,sampling_rate=SAMPLING_RATE)
        r_peak_indices = info.get('ECG_R_Peaks')
        rr_intervals = np.diff(r_peak_indices)/SAMPLING_RATE
        return rr_intervals
    except Exception as e:
        print(f"Error detecting R-peaks: {e}")
        return None

def extract_and_save_ecg_data(record_name, record_path, row, ecg_h5_files):
    """
    Extract ECG data for a specific AF episode and save it.
    Uses pre-calculated ECG data from H5 files.
    """
    try:
        start_file_idx = int(row['start_file_index'])
        end_file_idx = int(row['end_file_index'])
        start_qrs_idx = int(row['start_qrs_index'])
        end_qrs_idx = int(row['end_qrs_index'])
        af_duration = float(row['af_duration'])
        nsr_duration = float(row['nsr_before_duration'])
        
        # Load ECG data from the corresponding ECG H5 file
        if start_file_idx >= len(ecg_h5_files):
            return False
        
        ecg_file = ecg_h5_files[start_file_idx]
        ecg_path = os.path.join(record_path, ecg_file)
        ecg_data = load_h5_data(ecg_path)
        
        if ecg_data is None or len(ecg_data) == 0:
            return False
        
        # Extract the ECG segment for the episode
        sr_segment_start = start_qrs_idx - int(MIN_NSR_DURATION * SAMPLING_RATE)
        afib_segment_end = start_qrs_idx + int(MIN_AF_DURATION * SAMPLING_RATE)
        start_idx = max(0, sr_segment_start)
        end_idx = min(len(ecg_data), afib_segment_end)
        ecg_segment = ecg_data[start_idx:end_idx]

        
        if len(ecg_segment) == 0:
            return False
        
        rr_intervals = find_RR_intervals(ecg_segment)
        
        if len(rr_intervals) == 0:
            return False
        
        # Create output directory
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        
        # Save ECG segment as .npy file
        output_filename = f"{record_name}_af{af_duration:.0f}s_nsr{nsr_duration:.0f}s_rr.npy"
        output_filepath = os.path.join(OUTPUT_PATH, output_filename)
        
        np.save(output_filepath, rr_intervals)
        
        return True
    except Exception as e:
        print(f"    Error: {type(e).__name__}: {e}")
        return False

def extract_and_save_rr_data(record_name, record_path, row, rr_h5_files):
    """
    Extract RR data for a specific AF episode and save it.
    Uses pre-calculated RR intervals from H5 files.
    Converts ECG sample QRS indices to RR interval indices.
    """
    try:
        start_file_idx = int(row['start_file_index'])
        end_file_idx = int(row['end_file_index'])
        start_qrs_idx = int(row['start_qrs_index'])
        end_qrs_idx = int(row['end_qrs_index'])
        af_duration = float(row['af_duration'])
        nsr_duration = float(row['nsr_before_duration'])
        
        # Load RR data from the corresponding RR H5 file
        if start_file_idx >= len(rr_h5_files):
            return False
        
        rr_file = rr_h5_files[start_file_idx]
        rr_path = os.path.join(record_path, rr_file)
        rr_data = load_rr_data_from_h5(rr_path)
        
        if rr_data is None or len(rr_data) == 0:
            return False
        
        # Convert QRS sample indices to RR interval indices
        # With sampling rate of 200 Hz, approximately one QRS detection every 150-200 samples
        # We estimate ~150 samples per RR interval
        SAMPLES_PER_RR = 150
        
        rr_start_idx = start_qrs_idx // SAMPLES_PER_RR
        rr_end_idx = end_qrs_idx // SAMPLES_PER_RR
        
        # Ensure indices are within bounds
        rr_start_idx = max(0, min(rr_start_idx, len(rr_data) - 1))
        rr_end_idx = max(rr_start_idx + 1, min(rr_end_idx, len(rr_data)))
        
        # Extract the RR intervals for the episode
        rr_segment = rr_data[rr_start_idx:rr_end_idx]
        
        if len(rr_segment) == 0:
            return False
        
        # Create output directory
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        
        # Save RR intervals as .npy file
        output_filename = f"{record_name}_af{af_duration:.0f}s_nsr{nsr_duration:.0f}s_rr.npy"
        output_filepath = os.path.join(OUTPUT_PATH, output_filename)
        
        np.save(output_filepath, rr_segment)
        
        return True
    except Exception as e:
        print(f"    Error: {type(e).__name__}: {e}")
        return False

def filter_records():
    """Filter records based on AF duration and NSR duration criteria."""
    qualified_records = []
    
    print(f"Found {len(RECORDS)} records to process\n")
    
    for record_name in RECORDS:
        record_path = os.path.join(DATASET_PATH, record_name)
        
        # Check if it's a directory
        if not os.path.isdir(record_path):
            continue
        
        print(f"Processing record: {record_name}")
        
        # Find ECG labels CSV file
        csv_path = find_ecg_labels_csv(record_path)
        
        if csv_path is None:
            print(f"  ❌ No ECG labels CSV file found\n")
            continue
        
        print(f"  ✓ Found: {os.path.basename(csv_path)}")
        
        # Load the CSV file
        df = pd.read_csv(csv_path)
        if df is None:
            print(f"  ❌ Failed to load CSV\n")
            continue
        
        print(f"  ✓ Loaded {len(df)} records from CSV")
        
        # Filter based on criteria
        filtered = df[(df['af_duration'] >= MIN_AF_DURATION) & 
                      (df['nsr_before_duration'] >= MIN_NSR_DURATION)]
        
        if len(filtered) > 0:
            print(f"  ✓ Found {len(filtered)} qualifying entries")
            qualified_records.append({
                'record_name': record_name,
                'record_path': record_path,
                'csv_path': csv_path,
                'qualifying_entries': len(filtered),
                'data': filtered
            })
            print(f"    Details:")
            for idx, row in filtered.iterrows():
                print(f"      - AF duration: {row['af_duration']:.1f}s, NSR duration: {row['nsr_before_duration']:.1f}s")
        else:
            print(f"  ❌ No entries meet the criteria")
        
        print()
    
    return qualified_records

# Execute the analysis
if __name__ == "__main__":
    results = filter_records()
    
    print("\n" + "="*60)
    print(f"SUMMARY: {len(results)} records have qualifying data")
    print("="*60)
    
    print("\n\nExtracting and saving RR intervals...")
    print("="*60)
    
    total_saved = 0
    
    for result in tqdm(results, desc="Processing records"):
        record_name = result['record_name']
        record_path = result['record_path']
        data = result['data']
        
        # Find ECG H5 files for this record (to get pre-calculated RR intervals)
        ecg_h5_files = find_ecg_h5_files(record_path, record_name)
    
        
        if not ecg_h5_files:
            print(f"\n{record_name}: ❌ No ECG H5 files found (needed for RR extraction)")
            continue

        # Process each qualifying entry
        saved_count = 0
        for idx, row in tqdm(data.iterrows(), total=len(data), desc=f"  {record_name}", leave=False):
            if extract_and_save_ecg_data(record_name, record_path, row, ecg_h5_files):
                saved_count += 1
                total_saved += 1
        
        tqdm.write(f"{record_name}: {saved_count}/{len(data)} saved")
    
    print("\n" + "="*60)
    print(f"FINAL SUMMARY: {total_saved} RR data files saved to {OUTPUT_PATH}")
    print("="*60)

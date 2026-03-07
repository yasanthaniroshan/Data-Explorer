import os
import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def RMSSD(rr_intervals):
    diff = np.diff(rr_intervals)
    return np.sqrt(np.mean(diff**2))

def pNN50(rr_intervals):
    diff = np.diff(rr_intervals)
    return np.sum(np.abs(diff) > 0.05) / len(diff) * 100 # Since RR intervals in seconds, 50ms = 0.05s

def SDNN(rr_intervals):
    return np.std(rr_intervals)

def sample_entropy(rr_intervals, m=2, r=0.2):
    sample_entropy_, _ = nk.entropy_sample(rr_intervals, dimension=m, tolerance=r * np.std(rr_intervals))
    return sample_entropy_

def approximate_entropy(rr_intervals, m=2, r=0.2):
    approximate_entropy_, _ = nk.entropy_approximate(rr_intervals, dimension=m, tolerance=r * np.std(rr_intervals))
    return approximate_entropy_

def alpha_1(rr_intervals):
    alpha_1_, _ = nk.complexity_dfa(rr_intervals, order=1)
    return alpha_1_




PROCESSED_DATA_DIR = "/home/intellisense01/EML-Labs/datasets/Data-Explorer/processed_data_60min_nsr_5min_af"

window_length = 200 # 200 RR intervals
stride = 10 # 10 RR interval stride

rmssd_values = []
pnn50_values = []
alpha1_values = []
sample_entropy_values = []
approximate_entropy_values = []

data = []

list_of_files = os.listdir(PROCESSED_DATA_DIR)

for file in list_of_files:
    if file.endswith(".npy"):
        data.append(np.load(os.path.join(PROCESSED_DATA_DIR, file)))

min_length = min(len(d) for d in data)
data = [d[-min_length:] for d in data] # Truncate to the shortest length
data = np.array(data)
print(f"Number of records: {len(data)} | Length of each record: {data.shape[1]}")

number_of_windows = [(len(record) - window_length) // stride + 1 for record in data]

print(f"Total number of windows to process: {sum(number_of_windows)}")

pbar = tqdm(total=sum(number_of_windows), desc="Processing Records", unit="window")
for record in data:
    rmssd_record = []
    pnn50_record = []
    alpha_1_record = []
    sample_entropy_record = []
    approximate_entropy_record = []
    for start in range(0, len(record) - window_length + 1, stride):
        window = record[start:start+window_length]
        rmssd_record.append(RMSSD(window))
        pnn50_record.append(pNN50(window))
        alpha_1_record.append(alpha_1(window))
        sample_entropy_record.append(sample_entropy(window))
        approximate_entropy_record.append(approximate_entropy(window))
        pbar.update(1)  

    rmssd_values.append(rmssd_record)
    pnn50_values.append(pnn50_record)
    alpha1_values.append(alpha_1_record)
    sample_entropy_values.append(sample_entropy_record)
    approximate_entropy_values.append(approximate_entropy_record)

mean_rr = np.mean(data, axis=0)
std_rr = np.std(data, axis=0)
ci_95_rr = 1.96 * std_rr / np.sqrt(data.shape[0])


mean_rmssd = np.mean(rmssd_values, axis=0)
std_rmssd = np.std(rmssd_values, axis=0)
ci_95_rmssd = 1.96 * std_rmssd / np.sqrt(len(rmssd_values))

mean_pnn50 = np.mean(pnn50_values, axis=0)
std_pnn50 = np.std(pnn50_values, axis=0)
ci_95_pnn50 = 1.96 * std_pnn50 / np.sqrt(len(pnn50_values))

mean_alpha1 = np.mean(alpha1_values, axis=0)
std_alpha1 = np.std(alpha1_values, axis=0)
ci_95_alpha1 = 1.96 * std_alpha1 / np.sqrt(len(alpha1_values))

mean_sample_entropy = np.mean(sample_entropy_values, axis=0)
std_sample_entropy = np.std(sample_entropy_values, axis=0)
ci_95_sample_entropy = 1.96 * std_sample_entropy / np.sqrt(len(sample_entropy_values))

mean_approximate_entropy = np.mean(approximate_entropy_values, axis=0)
std_approximate_entropy = np.std(approximate_entropy_values, axis=0)
ci_95_approximate_entropy = 1.96 * std_approximate_entropy / np.sqrt(len(approximate_entropy_values))

plt.figure(figsize=(15,10))
plt.subplot(3,2,1)
plt.plot(mean_rr, label="Mean RR Interval")
plt.fill_between(range(len(mean_rr)), mean_rr - ci_95_rr, mean_rr + ci_95_rr, color='b', alpha=0.2, label="95% CI")
plt.title("Mean RR Interval with 95% Confidence Interval")
plt.xlabel("Time (s)")
plt.ylabel("RR Interval (s)")
plt.legend()
plt.grid()

plt.subplot(3,2,2)
plt.plot(mean_rmssd, label="Mean RMSSD")
plt.fill_between(range(len(mean_rmssd)), mean_rmssd - ci_95_rmssd, mean_rmssd + ci_95_rmssd, color='b', alpha=0.2, label="95% CI")
plt.title("Mean RMSSD with 95% Confidence Interval")
plt.xlabel("Time (s)")
plt.ylabel("RMSSD (s)")
plt.legend()
plt.grid()  

plt.subplot(3,2,3)
plt.plot(mean_pnn50, label="Mean pNN50")
plt.fill_between(range(len(mean_pnn50)), mean_pnn50 - ci_95_pnn50, mean_pnn50 + ci_95_pnn50, color='b', alpha=0.2, label="95% CI")
plt.title("Mean pNN50 with 95% Confidence Interval")
plt.xlabel("Time (s)")
plt.ylabel("pNN50 (%)")
plt.legend()
plt.grid()

plt.subplot(3,2,4)
plt.plot(mean_alpha1, label="Mean Alpha 1")
plt.fill_between(range(len(mean_alpha1)), mean_alpha1 - ci_95_alpha1, mean_alpha1 + ci_95_alpha1, color='b', alpha=0.2, label="95% CI")
plt.title("Mean Alpha 1 with 95% Confidence Interval")
plt.xlabel("Time (s)")
plt.ylabel("Alpha 1")
plt.legend()
plt.grid()

plt.subplot(3,2,5)
plt.plot(mean_sample_entropy, label="Mean Sample Entropy")
plt.fill_between(range(len(mean_sample_entropy)), mean_sample_entropy - ci_95_sample_entropy, mean_sample_entropy + ci_95_sample_entropy, color='b', alpha=0.2, label="95% CI")
plt.title("Mean Sample Entropy with 95% Confidence Interval")
plt.xlabel("Time (s)")
plt.ylabel("Sample Entropy")
plt.legend()
plt.grid()

plt.subplot(3,2,6)
plt.plot(mean_approximate_entropy, label="Mean Approximate Entropy")
plt.fill_between(range(len(mean_approximate_entropy)), mean_approximate_entropy - ci_95_approximate_entropy, mean_approximate_entropy + ci_95_approximate_entropy, color='b', alpha=0.2, label="95% CI")
plt.title("Mean Approximate Entropy with 95% Confidence Interval")
plt.xlabel("Time (s)")
plt.ylabel("Approximate Entropy")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("comparison_of_different_metrics.png", dpi=300, bbox_inches='tight')

print("Plots saved as comparison_of_different_metrics.png")
Background
Atrial fibrillation (AF) is a prevalent atrial arrhythmia that reduces quality of life and leads to complications such as embolic stroke and heart failure. Recent progress in machine learning and deep learning (DL) has demonstrated the potential to improve diagnostic accuracy significantly. Ensuring that DL models are robust and applicable across diverse factors such as ethnicity, age, and sex is crucial.

Despite the availability of several ECG databases to the research community, such as MITDB [1, 2], AFDB [2, 3], LTAFDB [2, 4], IRIDIA-AF [5], Icentia11k [2, 6, 7], and CPSC2021 [2, 8], none of them incorporate a sample from the Japanese population.

The SHDB-AF dataset was collected in previous work to evaluate the generalization performance of a DL algorithm for AF event detection from beat-to-beat time intervals, termed ArNet2 [9, 10], under different distribution shifts. A second DL model named RawECGNet [11] was developed and benchmarked against ArNet2. In contrast to ArNet2, RawECGNet was developed based on the raw, single-lead ECG signal. Regarding F1 score, ArNet2 achieved an F1 score of 0.92 for SHDB using beat-to-beat time intervals and 0.93 for RawECGNet using the raw ECG signal. Further results are presented in associated papers [10, 11].


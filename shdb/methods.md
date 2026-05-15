Methods
As previously mentioned, data provided in this project was initially collected for the purpose of evaluating generalization performance of a DL algorithm for AF detection denoted ArNet2, [9, 10]. Inclusion criteria and preprocessing steps applied to SHDB-AF, are described bellow.

Data collection
This database includes ECG recordings of adult subjects who underwent Holter monitoring as ordered by their treating physician from May 2019 to May 2023. Holters were recorded using Fukuda Holter monitor and digitized at 125Hz with two leads recorded, modified CC5 and NASA leads. Each recording lasts approximately 24 hours. While no reference beat annotations were extracted, each recording includes a diagnosis based on the free-text medical report prepared following the patient's examination.

Data preparation
Inclusion criteria
A total of 145 Holter recordings were collected, of which 128 were included in the current database. Seventeen recordings were excluded due to duplicates, missing clinical information, or low signal quality. Of the 128 included recordings, 98 were associated with 93 unique subjects and were annotated by certified cardiologists.

Following the methodology outlined in Biton et al. [10], approximately 100 recordings from each database were re-annotated by a cardiology fellow (MA). These subsets were stratified based on age, sex, and AF diagnosis. Specifically, 80 of these recordings came from subjects diagnosed with AF, as per the diagnosis recorded for each individual recording. Further details on the inclusion criteria for these selected recordings are provided in Biton et al. [10].

The remaining 30 recordings were included to complete the dataset, though rhythm annotations are not available for these recordings.

Preprocessing
Subsequently, all recordings underwent filtering using a zero-phase second-order infinite impulse response bandpass filter with a passband of [0.67 - 100] Hz [12] to eliminate baseline wander and high-frequency noise. Following this, the recordings were resampled to 200 Hz using an anti-aliasing filter. Beat annotations were then identified using the epltd implementation of the Pan and Tompkins algorithm [13].

Annotation protocol
A total of 98 recordings underwent manual beat-level annotation. To ensure patient confidentiality under HIPAA/GDPR regulations, cardiologists accessed a secure server remotely for the annotation process. Specifically, the PhysioZoo software [14, 15] was used to annotate the recordings. An annotation protocol was established, focusing on categorizing supraventricular arrhythmias including (1) AF, (2) Atrial Flutter, (3) Atrial tachycardia, and (4) other supraventricular tachycardias such as Wolf-Parkinson-White and intranodal tachycardias. Normal sinus rhythm and other rhythms were not annotated. MA spent an estimated average of 45 minutes per 24-hour Holter recording for annotation. A detailed description of the re-annotation protocol can be found in Biton et al. [10].

De-identification
Patient identifiers were anonymized following HIPAA guidelines and local regulations. Each participant was assigned a random 3-digit unique identifier ranging from 000 to 143. Numbers with fewer than three digits were padded with zeros to maintain consistent length. Dates directly associated with participants were randomly shifted by at least one year.
Data Description
ECG data
All ECG recordings are stored in the WFDB format. Each ECG file (.dat suffix) contains two channels: 'ECG1', representing modified CC5 leads, and 'ECG2', representing NASA leads.

Rhythm annotations
Rhythm annotations are provided at the beat level in separate annotation files (.atr). These files categorize rhythms into five types:

(AFIB (atrial fibrillation);
(AFL (atrial flutter);
(AT (atrial tachycardia);
(PAT (Other supraventricular tachycardias such as Wolf-Parkinson-White) and (NOD (intranodal tachycardias); and
(N (other, such as NSR, that were not labeled).
In the annotation files, rhythm annotations are listed under the 'aux_note' parameter. Each rhythm mark within the annotation files indicates the start of a rhythm interval. The appearance of a subsequent rhythm mark marks the beginning of a new interval.

The shortest interval lasts 2.5 seconds, while the longest extends to 24 hours, with a median duration of 47.5 seconds (Q2-Q3 range: 17.0 to 270.25 seconds).

Below are the counts for each annotated beat by rhythm and the overall intervals.

Mark	Rhythm label	Beats	Intervals
(N	Other	7,812,308	-
(AFIB	Atrial fibrillation	2,512,959	809
(AFL	Atrial flutter	195,659	45
(AT	Atrial tachycardia	48,800	57
(PAT and (NOD	Other supraventricular tachycardias	4,416	9
R-peak annotations are provided in files with the .qrs suffix.

Clinical data
The AdditionalData.csv file provides comprehensive clinical and demographic information for each subject, extracted from their Electronic Medical Records (EMR). The following table summarizes key characteristics of the dataset, offering an overview of patient demographics, medical history, and AF-related details. 

Variable	Mean ± SD / Median (IQR) / Count (%)
Total subjects	N = 122
Total Holter recordings	N = 128
Annotated recordings	N = 98
Age at Holter (years)	Mean ± SD: 68.0 ± 11.3
Sex	Female: 47 (38.5%), Male: 75 (61.4%)
Height (m)	Mean ± SD: 1.6 ± 0.1
Weight (kg)	Mean ± SD: 62.4 ± 13.9
BMI	Mean ± SD: 23.0 ± 4.1
AF Type	Paroxysmal (PAF): 80 (62.5%) , Persistent: 15 (11.7%) , Non-AF: 33 (25.8%)
Duration of AF (months)	Median (IQR): 9.0 [0.0, 36.0]
Previously Documented AFL	Yes: 17 (13.3%)
Previous AF ablation	Yes: 41 (32.0%)
Pacemaker presence	Yes: 1 (0.8%)
Anticoaogulation Use	YEs: 66 (51.5%)
Beta-Blocker Use (BB)	Yes: 39 (30.4%)
Non-BB Antiarrhythmic Drug Use	35 (27.3%)
Echocardiographic LAD (mm)	Mean ± SD: 39.6 ± 6.8
Echocardiographic LVEF (%)	Mean ± SD: 63.2 ± 11.6 
Comorbidities	CHF: 15 (11.7%), HTN: 43 (33.6%), DM: 8 (6.2%), Vascular Disease: 15 (11.7%)
Stroke History	Yes: 19 (11.7)
The dataset includes the following variables, providing further context on patient conditions and Holter recordings:

<Subject_ID>: The unique identifier for the subject.
<Data_ID>: A three-digit identifier assigned to each Holter recording.
<Annotated>: Indicates whether the Holter recording includes rhythm annotations.
<Height>: The patient's height as recorded in the EMR, in meters.
<Weight>: The patient's weight as recorded in the EMR, in kg.
<BMI>: The patient's Body Mass Index (BMI), calculated based on the height and weight in the EMR.
<Date_Holter>: The date on which the Holter recording was made.
<Holter_start_time>: Represents the relative time of day for each recording, formatted as hh:mm AM/PM, indicating the start time of the recording within a 24-hour period.
<Holter_recording_length>: Indicates the total duration of each recording, presented in the format hh:mm:ss.
<Indication_Holter>: The reason for the Holter recording, as documented in the medical report.
<Age_at_Holter>: The subject's age in years at the time of the Holter recording.
<Sex>: The subject's gender, categorized as either "male" or "female."
<AF_Type>: The final diagnosis of the subject based on the medical report following the Holter examination.
<Previously_Documented_AFL>: Indicates whether atrial flutter was documented prior to the Holter recording.
<Previous_AF_Ablation>: Indicates whether the subject had undergone catheter ablation for atrial fibrillation prior to the Holter examination.
<PPM_on_Holter>/<PPM_after_Holter>: Indicates whether the subject had a pacemaker implanted during or after the Holter recording.
<PPM_Indication>: The reason for pacemaker implantation, if applicable.
<PPM_Date>: The date on which the pacemaker was implanted, if applicable.
<Date_of_First_Diagnosis_of_AF_AFL>: The date when the subject was first diagnosed with atrial fibrillation or atrial flutter.
<AF_Duration_Months>: The duration of atrial fibrillation in months, from the first diagnosis of atrial fibrillation/atrial flutter until the Holter recording.
<Antiarrhythmic_Drug_nonBB>/<BB>/<Anticoagulation>: Lists any permanent medications taken by the subject prior to the Holter recording, including Beta-Blockers (BB), non-BB antiarrhythmic drugs, and anticoagulants.
<Date_of_1st_AF_Ablation>/<Ablation1_PVI>/<Ablation1_CTI>/<Ablation1_Others>/<Date_Redo_AF_Ablation>/<Redo_Detail>: Information regarding any atrial fibrillation ablation procedures, including the date of the first procedure, the type of ablation (PVI, CTI, or others), and details on any redo ablations.
<Echo_Date>/<Echo_LAD>/<Echo_LVEF>/<Echo_LV_Asynergy>: Details about any echocardiogram performed on the subject, including the date of the echo, left atrial diameter (LAD), left ventricular ejection fraction (LVEF), and left ventricular asynergy.
<Moderate_or_Severe_MR>/<Moderate_or_Severe_TR>/<Moderate_or_Severe_AS>/<Moderate_or_Severe_AR>: Indication of moderate and severe heart valve conditions present in the subject, including Mitral Regurgitation (MR), Tricuspid Regurgitation (TR), Aortic Stenosis (AS), and Aortic Regurgitation (AR).
<CHF>/<HTN>/<DM>/<Vascular_Diseases>: Indications of relevant comorbidities, including Congestive Heart Failure (CHF), Hypertension (HTN), Diabetes Mellitus (DM), and Vascular Disease (Vascular_Diseases).
<Stroke>: Indicates if the subject had a stroke prior to the Holter recording.
<Comments>: Miscellaneous textual information not covered in the other columns.
This additional dataset helps to enrich the analysis by providing crucial context on each subject's medical background, diagnosis, and treatment history, facilitating more detailed research and insights from the Holter recording data.

Each file name corresponds uniquely to a Data_ID, specifically matching the 'Data_ID' field in the AdditionalData.csv.

The file included, example.png, provides an illustration of an ECG example.


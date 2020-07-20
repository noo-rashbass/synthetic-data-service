import pandas as pd
import numpy as np
from datetime import date, timedelta

# reading in and assigning values
data = pd.read_csv('isaFull.tsv', '\t')
data['Visit date [EUPATH_0000091]'] = pd.to_datetime(data['Visit date [EUPATH_0000091]'])

data = data.assign(real=True)
imputed_df = data[0:0]

minmax_time = [min(data['Visit date [EUPATH_0000091]']),\
               max(data['Visit date [EUPATH_0000091]'])]

# latest end date
series_end_date = minmax_time[1]

delta = timedelta(days=1)
patients = data['Participant_Id'].unique()

# imputing for each patient
for patient_id in patients:

    patient = data[data['Participant_Id'] == patient_id].\
                        sort_values(['Visit date [EUPATH_0000091]']).reset_index(drop=True)

    # earliest start date
    series_start_date = minmax_time[0]
    
    minmax_time_patient = [min(patient['Visit date [EUPATH_0000091]']),\
                            max(patient['Visit date [EUPATH_0000091]'])]
    
    patient_start_date = minmax_time_patient[0]
    patient_end_date = minmax_time_patient[1]

    day_difference = (patient_start_date - series_start_date).days

    observation_id = patient.iloc[0]['Observation_Id']
    imputed_obs_id = observation_id - day_difference
    age = patient.iloc[0]["Age at visit (years) [EUPATH_0000113]"]
    current_age = age - day_difference/365
    current_date = series_start_date

    # imputing from series start date to patient start date
    while current_date < patient_start_date:
        
        abd_pain_duration = patient.iloc[0]['Abdominal pain duration (days) [EUPATH_0000154]']
        if abd_pain_duration - day_difference < 0:
            abd_pain = "Unable to assess"
            abd_pain_duration = 0
        else:
            abd_pain = "Yes"
            abd_pain_duration -= day_difference     # make int when writing

        admitting_hospital = np.nan

        anorexia_duration = patient.iloc[0]['Anorexia duration (days) [EUPATH_0000155]']
        if anorexia_duration - day_difference < 0:
            anorexia = "Unable to assess"
            anorexia_duration = 0
        else:
            anorexia = "Yes"
            anorexia_duration -= day_difference     # make int when writing

        asex_plasmod_density = np.nan
        asex_plasmod_present = np.nan
        complex_diagnosis_basis = np.nan
        complicated_malaria = np.nan

        cough_duration = patient.iloc[0]['Cough duration (days) [EUPATH_0000156]']
        if cough_duration - day_difference < 0:
            cough = "Unable to assess"
            cough_duration = 0
        else:
            cough = "Yes"
            cough_duration -= day_difference     # make int when writing

        days_since_enrollment = np.nan
        diagnosis_at_hospital = np.nan
        

        imputed_obs_id += 1
        current_age += 1/365

        current_date += delta
        day_difference -= 1


    break

'''
column = "Asexual Plasmodium parasite density, by microscopy [EUPATH_0000092]"
diagnosis = data[["Observation_Id", "Participant_Id", column, 'Visit date [EUPATH_0000091]']]
print(diagnosis)
print(patient[["Observation_Id", "Participant_Id", column, 'Visit date [EUPATH_0000091]']])

print(diagnosis[diagnosis[column].isnull()])
print(diagnosis[column].unique())
'''
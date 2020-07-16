import pandas as pd
from datetime import date, timedelta

data = pd.read_csv('isaFull.tsv', '\t')
data['Visit date [EUPATH_0000091]'] = pd.to_datetime(data['Visit date [EUPATH_0000091]'])

patient = data[data['Participant_Id'] == 1001].\
                        sort_values(['Visit date [EUPATH_0000091]']).reset_index(drop=True)

minmax_time = [min(data['Visit date [EUPATH_0000091]']),\
               max(data['Visit date [EUPATH_0000091]'])]

minmax_time_patient = [min(patient['Visit date [EUPATH_0000091]']),\
                        max(patient['Visit date [EUPATH_0000091]'])]

# earliest start date and latest end date
series_start_date = minmax_time[0]
series_end_date = minmax_time[1]

patient_start_date = minmax_time_patient[0]
patient_end_date = minmax_time_patient[1]

anorexia = data[data["Anorexia [SYMP_0000523]"] == "Unable to assess"]

age_temp_patient = patient[["Observation_Id", "Age at visit (years) [EUPATH_0000113]",\
                            "Temperature (C) [EUPATH_0000110]", "Visit date [EUPATH_0000091]"]]

'''Imputing backwards to series start date'''
delta = timedelta(days=1)
day_difference = (patient_start_date - series_start_date).days

# observation ID
observation_id = patient.iloc[0]['Observation_Id']
imputed_obs_id = observation_id - day_difference

# participant ID and household ID stay the same
# abdominal pain - 'unable to assess', or yes if from future visit
# abdominal pain duration - 0 if unable to assess, compute from future visit duration otherwise
# admitting hospital - na
# age imputation below
# anorexia - 'unable to assess', or yes if from future visit
# 

age = patient.iloc[0]["Age at visit (years) [EUPATH_0000113]"]

series_start_age = age - day_difference/365

# imputing from earliest start data in dataset to first patient visit
while series_start_date < patient_start_date:
    series_start_date += delta
    series_start_age += 1/365

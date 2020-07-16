import pandas as pd
from datetime import date, timedelta

data = pd.read_csv('isaFull.tsv', '\t')
data['Visit date [EUPATH_0000091]'] = pd.to_datetime(data['Visit date [EUPATH_0000091]'])

patient = data[data['Participant_Id']==1007].\
                        sort_values(['Visit date [EUPATH_0000091]']).reset_index(drop=True)

minmax_time = [min(data['Visit date [EUPATH_0000091]']),\
               max(data['Visit date [EUPATH_0000091]'])]

minmax_time_patient = [min(patient['Visit date [EUPATH_0000091]']),\
                        max(patient['Visit date [EUPATH_0000091]'])]

series_start_date = minmax_time[0]
series_end_date = minmax_time[1]

patient_start_date = minmax_time_patient[0]
patient_end_date = minmax_time_patient[1]

age_temp_patient = patient[["Age at visit (years) [EUPATH_0000113]",\
                            "Temperature (C) [EUPATH_0000110]", "Visit date [EUPATH_0000091]"]]

delta = timedelta(days=1)

first_visit = age_temp_patient.iloc[0]["Visit date [EUPATH_0000091]"]
second_visit = age_temp_patient.iloc[1]["Visit date [EUPATH_0000091]"]

days = 0
age = patient.iloc[0]["Age at visit (years) [EUPATH_0000113]"]
first_temperature = patient.iloc[0]["Temperature (C) [EUPATH_0000110]"]
second_temperature = patient.iloc[1]["Temperature (C) [EUPATH_0000110]"]

print(minmax_time)
print(minmax_time_patient)

while first_visit < second_visit:
    print(first_temperature, second_temperature)
    first_visit += delta
    days += 1
    age += 1/365

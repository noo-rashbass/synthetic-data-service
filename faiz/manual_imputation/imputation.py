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

    print(patient)
    # earliest start date
    series_start_date = minmax_time[0]
    
    minmax_time_patient = [min(patient['Visit date [EUPATH_0000091]']),\
                            max(patient['Visit date [EUPATH_0000091]'])]
    
    patient_start_date = minmax_time_patient[0]
    patient_end_date = minmax_time_patient[1]

    day_difference = (patient_start_date - series_start_date).days

    observation_id = patient.iloc[0]['Observation_Id']
    imputed_obs_id = observation_id - day_difference
    household_id = patient.iloc[0]['Household_Id']
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

        diarrhoea_duration = patient.iloc[0]['Diarrhea duration (days) [EUPATH_0000157]']
        if diarrhoea_duration - day_difference < 0:
            diarrhoea = "Unable to assess"
            diarrhoea_duration = 0
        else:
            diarrhoea = "Yes"
            diarrhoea_duration -= day_difference     # make int when writing
        
        fatigue_duration = patient.iloc[0]['Fatigue duration (days) [EUPATH_0000158]']
        if fatigue_duration - day_difference < 0:
            fatigue = "Unable to assess"
            fatigue_duration = 0
        else:
            fatigue = "Yes"
            fatigue_duration -= day_difference     # make int when writing
        
        febrile_duration = patient.iloc[0]['Fever, subjective duration (days) [EUPATH_0000164]']
        if febrile_duration - day_difference < 0:
            febrile = "Unable to assess"
            febrile_duration = 0
        else:
            febrile = "Yes"
            febrile_duration -= day_difference     # make int when writing

        headache_duration = patient.iloc[0]['Headache duration (days) [EUPATH_0000159]']
        if headache_duration - day_difference < 0:
            headache = "Unable to assess"
            headache_duration = 0
        else:
            headache = "Yes"
            headache_duration -= day_difference     # make int when writing
        
        # set height as NAN for children and constant for adults (may change for children in the future)
        height = np.nan
        if current_age > 20:
            height = patient.iloc[0]['Height (cm) [EUPATH_0010075]']

        # averaging haemoglobin
        haemoglobin = round(patient['Hemoglobin (g/dL) [EUPATH_0000047]'].mean(), 1)
        hospital_admission_date = np.nan
        hospital_discharge_date = np.nan
        itn = np.nan

        jaundice_duration = patient.iloc[0]['Jaundice duration (days) [EUPATH_0000160]']
        if jaundice_duration - day_difference < 0:
            jaundice = "Unable to assess"
            jaundice_duration = 0
        else:
            jaundice = "Yes"
            jaundice_duration -= day_difference     # make int when writing

        joint_pains_duration = patient.iloc[0]['Joint pains duration (days) [EUPATH_0000161]']
        if joint_pains_duration - day_difference < 0:
            joint_pains = "Unable to assess"
            joint_pains_duration = 0
        else:
            joint_pains = "Yes"
            joint_pains_duration -= day_difference     # make int when writing

        malaria_diagnosis = np.nan
        malaria_diagnosis_parasite_status = "Blood smear not indicated"
        malaria_treatment = "No malaria medications given"

        muscle_aches_duration = patient.iloc[0]['Muscle aches duration (days) [EUPATH_0000162]']
        if muscle_aches_duration - day_difference < 0:
            muscle_aches = "Unable to assess"
            muscle_aches_duration = 0
        else:
            muscle_aches = "Yes"
            muscle_aches_duration -= day_difference     # make int when writing

        non_malaria_medication = np.nan
        other_diagnosis = np.nan
        other_medical_complaint = np.nan
        plasmod_gametocytes_present = np.nan

        seizures_duration = patient.iloc[0]['Seizures duration (days) [EUPATH_0000163]']
        if seizures_duration - day_difference < 0:
            seizures = "Unable to assess"
            seizures_duration = 0
        else:
            seizures = "Yes"
            seizures_duration -= day_difference     # make int when writing
        
        severe_malaria_criteria = np.nan
        subjective_fever = np.nan
        submic_plasmod_present = np.nan

        temperature = round(patient['Temperature (C) [EUPATH_0000110]'].mean(), 1)
        visit_date = current_date
        visit_type = "Scheduled visit"

        vomiting_duration = patient.iloc[0]['Seizures duration (days) [EUPATH_0000163]']
        if vomiting_duration - day_difference < 0:
            vomiting = "Unable to assess"
            vomiting_duration = 0
        else:
            vomiting = "Yes"
            vomiting_duration -= day_difference     # make int when writing

        # weight NAN for children, averaging for adults
        weight = np.nan
        if current_age > 20:
            weight = round(patient['Weight (kg) [EUPATH_0000732]'].mean(), 1)
        
        imputed_row = {
            "Observation_Id": imputed_obs_id, 
            "Participant_Id": patient_id,
            "Household_Id": household_id,
            "Abdominal pain [HP_0002027]": abd_pain,
            "Abdominal pain duration (days) [EUPATH_0000154]": abd_pain_duration,
            "Admitting hospital [EUPATH_0000318]": admitting_hospital,
            "Age at visit (years) [EUPATH_0000113]": round(current_age, 2),
            "Anorexia [SYMP_0000523]": anorexia,
            "Anorexia duration (days) [EUPATH_0000155]": anorexia_duration,
            "Asexual Plasmodium parasite density, by microscopy [EUPATH_0000092]": asex_plasmod_density,
            "Asexual Plasmodium parasites present, by microscopy [EUPATH_0000048]": asex_plasmod_present,
            "Basis of complicated diagnosis [EUPATH_0000316]": complex_diagnosis_basis,
            "Complicated malaria [EUPATH_0000040]": complicated_malaria,
            "Cough [SYMP_0000614]": cough,
            "Cough duration (days) [EUPATH_0000156]": cough_duration,
            "Days since enrollment [EUPATH_0000191]": days_since_enrollment,
            "Diagnosis at hospitalization [EUPATH_0000638]": diagnosis_at_hospital,
            "Diarrhea [DOID_13250]": diarrhoea,
            "Diarrhea duration (days) [EUPATH_0000157]": diarrhoea_duration,
            "Fatigue [SYMP_0019177]": fatigue,
            "Fatigue duration (days) [EUPATH_0000158]": fatigue_duration,
            "Febrile [EUPATH_0000097]": febrile,
            "Fever, subjective duration (days) [EUPATH_0000164]": febrile_duration,
            "Headache [HP_0002315]": headache,
            "Headache duration (days) [EUPATH_0000159]": headache_duration,
            "Height (cm) [EUPATH_0010075]": height,
            "Hemoglobin (g/dL) [EUPATH_0000047]": haemoglobin,
            "Hospital admission date [EUPATH_0000319]": hospital_admission_date,
            "Hospital discharge date [EUPATH_0000320]": hospital_discharge_date,
            "ITN last night [EUPATH_0000216]": itn,
            "Jaundice [HP_0000952]": jaundice,
            "Jaundice duration (days) [EUPATH_0000160]": jaundice_duration,
            "Joint pains [SYMP_0000064]": joint_pains,
            "Joint pains duration (days) [EUPATH_0000161]": joint_pains_duration,
            "Malaria diagnosis [EUPATH_0000090]": malaria_diagnosis,
            "Malaria diagnosis and parasite status [EUPATH_0000338]": malaria_diagnosis_parasite_status,
            "Malaria treatment [EUPATH_0000740]": malaria_treatment,
            "Muscle aches [EUPATH_0000252]": muscle_aches,
            "Muscle aches duration (days) [EUPATH_0000162]": muscle_aches_duration,
            "Non-malaria medication [EUPATH_0000059]": non_malaria_medication,
            "Other diagnosis [EUPATH_0000317]": other_diagnosis,
            "Other medical complaint [EUPATH_0020002]": other_medical_complaint,
            "Plasmodium gametocytes present, by microscopy [EUPATH_0000207]": plasmod_gametocytes_present,
            "Seizures [SYMP_0000124]": seizures,
            "Seizures duration (days) [EUPATH_0000163]": seizures_duration,
            "Severe malaria criteria [EUPATH_0000046]": severe_malaria_criteria,
            "Subjective fever [EUPATH_0000100]": subjective_fever,
            "Submicroscopic Plasmodium present, by LAMP [EUPATH_0000487]": submic_plasmod_present,
            "Temperature (C) [EUPATH_0000110]": temperature,
            "Visit date [EUPATH_0000091]": visit_date,
            "Visit type [EUPATH_0000311]": visit_type,
            "Vomiting [HP_0002013]": vomiting,
            "Vomiting duration (days) [EUPATH_0000165]": vomiting_duration,
            "Weight (kg) [EUPATH_0000732]": weight,
            "real": False
        }

        patient = patient.append(imputed_row, ignore_index=True)

        imputed_obs_id += 1
        current_age += 1/365

        current_date += delta
        day_difference -= 1

    print(patient[['Observation_Id', 'Participant_Id', 'Visit date [EUPATH_0000091]']].\
            sort_values(['Visit date [EUPATH_0000091]']))
    break

'''
column = "Height (cm) [EUPATH_0010075]"
diagnosis = data[["Observation_Id", "Participant_Id", column, 'Visit date [EUPATH_0000091]']]
print(diagnosis)
print(patient[["Observation_Id", "Participant_Id", column, 'Visit date [EUPATH_0000091]']])

print(diagnosis[diagnosis[column].isnull()])
print(diagnosis[column].unique())
'''
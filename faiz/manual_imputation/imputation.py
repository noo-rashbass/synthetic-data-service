import pandas as pd
import numpy as np
import time
from datetime import date, timedelta

start_time = time.time()

# reading in and assigning values
data = pd.read_csv('isaFull.tsv', '\t')

age_string = 'Age at visit (years) [EUPATH_0000113]'
height_string = 'Height (cm) [EUPATH_0010075]'
haemoglobin_string = 'Hemoglobin (g/dL) [EUPATH_0000047]'
temperature_string = 'Temperature (C) [EUPATH_0000110]'
visit_date_string = 'Visit date [EUPATH_0000091]'
visit_type = "Scheduled visit"
weight_string = 'Weight (kg) [EUPATH_0000732]'

data[visit_date_string] = pd.to_datetime(data[visit_date_string])

data = data.assign(real=1000)
imputed_df = data[0:0]

minmax_time = [min(data[visit_date_string]),\
               max(data[visit_date_string])]

# setting nan and constant values
admitting_hospital = np.nan
ax_plas_density = np.nan
ax_plas_present = np.nan
complex_diagnosis_basis = np.nan
complicated_malaria = np.nan
days_since_enrollment = np.nan
diagnosis_at_hospital = np.nan
hospital_admission_date = np.nan
hospital_discharge_date = np.nan
itn = np.nan
malaria_diagnosis = np.nan
non_malaria_medication = np.nan
other_diagnosis = np.nan
other_medical_complaint = np.nan
plas_gam_present = np.nan
severe_malaria_criteria = np.nan
subjective_fever = np.nan
submic_plas_present = np.nan

malaria_diagnosis_parasite = "Blood smear not indicated"
malaria_treatment = "No malaria medications given"

# latest end date
series_end_date = minmax_time[1]

delta = timedelta(days=1)
patients = data['Participant_Id'].unique()


# calculates whether ailment is currently active based on future trip to hospital
def duration_calculator(duration, day_difference):

    if duration - day_difference < 0:
        status = "Unable to assess"
        duration = 0
    else:
        status = "Yes"
        duration -= day_difference     # make int when writing

    return status, str(duration)


# checks whether column has same values for all rows, e.g. height
def is_equal_throughout(column):

    a = column.to_numpy()
    return (a[0] == a).all()


# linearly model a variable between visits
# returns tuple (boolean, value) - boolean if it can be interpolated (True if no nan)
# value is daily change or mean column value depending on boolean
def get_linear_modelling_value(patient, real_visit, day_difference, column):

    current_value = patient.iloc[real_visit - 1][column]
    new_value = patient.iloc[real_visit][column]

    if pd.isna(current_value) or pd.isna(new_value):
        return (False, patient[column].mean())
    else:
        if current_value != new_value:
            daily_change = (new_value - current_value)/day_difference
            return (True, daily_change)
        return (True, 0)


def apply_linear_modelling(variable, change):

    if change[0]:
        variable += change[1]
    else:
        variable = change[1]   # take the mean

    return variable


# imputing for each patient
for patient_id in patients:

    patient = data[data['Participant_Id'] == patient_id].\
                        sort_values([visit_date_string]).reset_index(drop=True)
    imputed_patient = patient

    # earliest start date
    series_start_date = minmax_time[0]

    minmax_time_patient = [min(patient[visit_date_string]),\
                            max(patient[visit_date_string])]

    patient_start_date = minmax_time_patient[0]
    patient_end_date = minmax_time_patient[1]

    # Round 1
    day_difference = (patient_start_date - series_start_date).days

    first_row = patient.iloc[0]
    last_row = patient.iloc[-1]

    observation_id = first_row['Observation_Id']
    imputed_obs_id = observation_id - day_difference
    household_id = first_row['Household_Id']
    age = first_row[age_string]
    current_age = age - day_difference/365
    current_date = series_start_date
    same_height_throughout = is_equal_throughout(patient[height_string])

    # averaging values
    average_haemoglobin = round(patient[haemoglobin_string].mean(), 1)
    average_temperature = round(patient[temperature_string].mean(), 1)
    average_weight = round(patient[weight_string].mean() * 2) / 2

    # set height as NAN for children and constant for adults (may change for children in the future)
    height = np.nan
    if same_height_throughout:
        height = first_row[height_string]

    # imputing from series start date to patient start date
    while current_date < patient_start_date:

        abd_pain_duration = first_row['Abdominal pain duration (days) [EUPATH_0000154]']
        abd_pain, abd_pain_duration = duration_calculator(abd_pain_duration, day_difference)

        anorexia_duration = first_row['Anorexia duration (days) [EUPATH_0000155]']
        anorexia, anorexia_duration = duration_calculator(anorexia_duration, day_difference)

        cough_duration = first_row['Cough duration (days) [EUPATH_0000156]']
        cough, cough_duration = duration_calculator(cough_duration, day_difference)

        diarrhoea_duration = first_row['Diarrhea duration (days) [EUPATH_0000157]']
        diarrhoea, diarrhoea_duration = duration_calculator(diarrhoea_duration, day_difference)

        fatigue_duration = first_row['Fatigue duration (days) [EUPATH_0000158]']
        fatigue, fatigue_duration = duration_calculator(fatigue_duration, day_difference)

        febrile_duration = first_row['Fever, subjective duration (days) [EUPATH_0000164]']
        febrile, febrile_duration = duration_calculator(febrile_duration, day_difference)

        headache_duration = first_row['Headache duration (days) [EUPATH_0000159]']
        headache, headache_duration = duration_calculator(headache_duration, day_difference)

        jaundice_duration = first_row['Jaundice duration (days) [EUPATH_0000160]']
        jaundice, jaundice_duration = duration_calculator(jaundice_duration, day_difference)

        joint_pains_duration = first_row['Joint pains duration (days) [EUPATH_0000161]']
        joint_pains, joint_pains_duration = duration_calculator(joint_pains_duration, day_difference)

        muscle_aches_duration = first_row['Muscle aches duration (days) [EUPATH_0000162]']
        muscle_aches, muscle_aches_duration = duration_calculator(muscle_aches_duration, day_difference)

        seizures_duration = first_row['Seizures duration (days) [EUPATH_0000163]']
        seizures, seizures_duration = duration_calculator(seizures_duration, day_difference)

        vomiting_duration = first_row['Seizures duration (days) [EUPATH_0000163]']
        vomiting, vomiting_duration = duration_calculator(vomiting_duration, day_difference)

        # weight NAN for children, averaging for adults
        weight = np.nan
        if current_age > 20:
            weight = average_weight

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
            "Asexual Plasmodium parasite density, by microscopy [EUPATH_0000092]": ax_plas_density,
            "Asexual Plasmodium parasites present, by microscopy [EUPATH_0000048]": ax_plas_present,
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
            "Height (cm) [EUPATH_0010075]": str(int(height)),
            "Hemoglobin (g/dL) [EUPATH_0000047]": average_haemoglobin,
            "Hospital admission date [EUPATH_0000319]": hospital_admission_date,
            "Hospital discharge date [EUPATH_0000320]": hospital_discharge_date,
            "ITN last night [EUPATH_0000216]": itn,
            "Jaundice [HP_0000952]": jaundice,
            "Jaundice duration (days) [EUPATH_0000160]": jaundice_duration,
            "Joint pains [SYMP_0000064]": joint_pains,
            "Joint pains duration (days) [EUPATH_0000161]": joint_pains_duration,
            "Malaria diagnosis [EUPATH_0000090]": malaria_diagnosis,
            "Malaria diagnosis and parasite status [EUPATH_0000338]": malaria_diagnosis_parasite,
            "Malaria treatment [EUPATH_0000740]": malaria_treatment,
            "Muscle aches [EUPATH_0000252]": muscle_aches,
            "Muscle aches duration (days) [EUPATH_0000162]": muscle_aches_duration,
            "Non-malaria medication [EUPATH_0000059]": non_malaria_medication,
            "Other diagnosis [EUPATH_0000317]": other_diagnosis,
            "Other medical complaint [EUPATH_0020002]": other_medical_complaint,
            "Plasmodium gametocytes present, by microscopy [EUPATH_0000207]": plas_gam_present,
            "Seizures [SYMP_0000124]": seizures,
            "Seizures duration (days) [EUPATH_0000163]": seizures_duration,
            "Severe malaria criteria [EUPATH_0000046]": severe_malaria_criteria,
            "Subjective fever [EUPATH_0000100]": subjective_fever,
            "Submicroscopic Plasmodium present, by LAMP [EUPATH_0000487]": submic_plas_present,
            "Temperature (C) [EUPATH_0000110]": average_temperature,
            "Visit date [EUPATH_0000091]": current_date,
            "Visit type [EUPATH_0000311]": visit_type,
            "Vomiting [HP_0002013]": vomiting,
            "Vomiting duration (days) [EUPATH_0000165]": vomiting_duration,
            "Weight (kg) [EUPATH_0000732]": weight,
            "real": 999
        }

        # append imputed row to patient
        imputed_patient = imputed_patient.append(imputed_row, ignore_index=True)
        imputed_obs_id += 1
        current_age += 1/365

        current_date += delta
        day_difference -= 1

# =====================================================================================================

    # Round 2
    # imputation between visit dates
    visit_dates = patient[visit_date_string]
    current_date += delta
    real_visit = 1
    day_difference = (visit_dates.iloc[real_visit] - current_date).days

    current_age = first_row[age_string] + 1/365 
    current_height = first_row[height_string]
    current_haemoglobin = first_row[haemoglobin_string]
    current_temperature = first_row[temperature_string]
    current_weight = first_row[weight_string]

    daily_height_increase = get_linear_modelling_value(patient, 1, day_difference, height_string)
    daily_haemoglobin_change =\
        get_linear_modelling_value(patient, 1, day_difference, haemoglobin_string)
    daily_temperature_change =\
        get_linear_modelling_value(patient, 1, day_difference, temperature_string)
    daily_weight_change = get_linear_modelling_value(patient, 1, day_difference, weight_string)

    # go through each date from first visit to last visit and check if real visit exists on this date
    # if not, impute one
    while current_date < patient_end_date:

        if current_date == visit_dates.iloc[real_visit]:

            # reset values to current
            current_age = patient.iloc[real_visit][age_string]
            current_height = patient.iloc[real_visit][height_string]
            current_haemoglobin = patient.iloc[real_visit][haemoglobin_string]
            current_temperature = patient.iloc[real_visit][temperature_string]
            current_weight = patient.iloc[real_visit][weight_string]

            # increment time tracker variables
            real_visit += 1
            day_difference = (visit_dates.iloc[real_visit] - current_date).days

            # check if same values between these visits & linearly model values between visits
            daily_height_increase =\
                get_linear_modelling_value(patient, real_visit, day_difference, height_string)
            daily_haemoglobin_change =\
                get_linear_modelling_value(patient, real_visit, day_difference, haemoglobin_string)
            daily_temperature_change =\
                get_linear_modelling_value(patient, real_visit, day_difference, temperature_string)
            daily_weight_change =\
                get_linear_modelling_value(patient, real_visit, day_difference, weight_string)

        else:

            # do imputation for this date
            abd_pain_duration =\
                patient.iloc[real_visit]['Abdominal pain duration (days) [EUPATH_0000154]']
            abd_pain, abd_pain_duration = duration_calculator(abd_pain_duration, day_difference)

            anorexia_duration = patient.iloc[real_visit]['Anorexia duration (days) [EUPATH_0000155]']
            anorexia, anorexia_duration = duration_calculator(anorexia_duration, day_difference)

            cough_duration = patient.iloc[real_visit]['Cough duration (days) [EUPATH_0000156]']
            cough, cough_duration = duration_calculator(cough_duration, day_difference)

            days_since_enrollment = (current_date - patient_start_date).days

            diarrhoea_duration = patient.iloc[real_visit]['Diarrhea duration (days) [EUPATH_0000157]']
            diarrhoea, diarrhoea_duration = duration_calculator(diarrhoea_duration, day_difference)

            fatigue_duration = patient.iloc[real_visit]['Fatigue duration (days) [EUPATH_0000158]']
            fatigue, fatigue_duration = duration_calculator(fatigue_duration, day_difference)

            febrile_duration =\
                patient.iloc[real_visit]['Fever, subjective duration (days) [EUPATH_0000164]']
            febrile, febrile_duration = duration_calculator(febrile_duration, day_difference)

            headache_duration = patient.iloc[real_visit]['Headache duration (days) [EUPATH_0000159]']
            headache, headache_duration = duration_calculator(headache_duration, day_difference)

            current_height = apply_linear_modelling(current_height, daily_height_increase)
            current_haemoglobin = apply_linear_modelling(current_haemoglobin, daily_haemoglobin_change)

            jaundice_duration = patient.iloc[real_visit]['Jaundice duration (days) [EUPATH_0000160]']
            jaundice, jaundice_duration = duration_calculator(jaundice_duration, day_difference)

            joint_pains_duration =\
                patient.iloc[real_visit]['Joint pains duration (days) [EUPATH_0000161]']
            joint_pains, joint_pains_duration =\
                duration_calculator(joint_pains_duration, day_difference)

            muscle_aches_duration =\
                patient.iloc[real_visit]['Muscle aches duration (days) [EUPATH_0000162]']
            muscle_aches, muscle_aches_duration =\
                duration_calculator(muscle_aches_duration, day_difference)

            seizures_duration = patient.iloc[real_visit]['Seizures duration (days) [EUPATH_0000163]']
            seizures, seizures_duration = duration_calculator(seizures_duration, day_difference)

            current_temperature = apply_linear_modelling(current_temperature, daily_temperature_change)

            vomiting_duration = patient.iloc[real_visit]['Seizures duration (days) [EUPATH_0000163]']
            vomiting, vomiting_duration = duration_calculator(vomiting_duration, day_difference)

            current_weight = apply_linear_modelling(current_weight, daily_weight_change)

            imputed_row = {
                "Observation_Id": observation_id, 
                "Participant_Id": patient_id,
                "Household_Id": household_id,
                "Abdominal pain [HP_0002027]": abd_pain,
                "Abdominal pain duration (days) [EUPATH_0000154]": abd_pain_duration,
                "Admitting hospital [EUPATH_0000318]": admitting_hospital,
                "Age at visit (years) [EUPATH_0000113]": round(current_age, 2),
                "Anorexia [SYMP_0000523]": anorexia,
                "Anorexia duration (days) [EUPATH_0000155]": anorexia_duration,
                "Asexual Plasmodium parasite density, by microscopy [EUPATH_0000092]": ax_plas_density,
                "Asexual Plasmodium parasites present, by microscopy [EUPATH_0000048]": ax_plas_present,
                "Basis of complicated diagnosis [EUPATH_0000316]": complex_diagnosis_basis,
                "Complicated malaria [EUPATH_0000040]": complicated_malaria,
                "Cough [SYMP_0000614]": cough,
                "Cough duration (days) [EUPATH_0000156]": cough_duration,
                "Days since enrollment [EUPATH_0000191]": str(days_since_enrollment),
                "Diagnosis at hospitalization [EUPATH_0000638]": diagnosis_at_hospital,
                "Diarrhea [DOID_13250]": diarrhoea,
                "Diarrhea duration (days) [EUPATH_0000157]": diarrhoea_duration,
                "Fatigue [SYMP_0019177]": fatigue,
                "Fatigue duration (days) [EUPATH_0000158]": fatigue_duration,
                "Febrile [EUPATH_0000097]": febrile,
                "Fever, subjective duration (days) [EUPATH_0000164]": febrile_duration,
                "Headache [HP_0002315]": headache,
                "Headache duration (days) [EUPATH_0000159]": headache_duration,
                "Height (cm) [EUPATH_0010075]": str(int(current_height)),
                "Hemoglobin (g/dL) [EUPATH_0000047]": round(current_haemoglobin, 1),
                "Hospital admission date [EUPATH_0000319]": hospital_admission_date,
                "Hospital discharge date [EUPATH_0000320]": hospital_discharge_date,
                "ITN last night [EUPATH_0000216]": itn,
                "Jaundice [HP_0000952]": jaundice,
                "Jaundice duration (days) [EUPATH_0000160]": jaundice_duration,
                "Joint pains [SYMP_0000064]": joint_pains,
                "Joint pains duration (days) [EUPATH_0000161]": joint_pains_duration,
                "Malaria diagnosis [EUPATH_0000090]": malaria_diagnosis,
                "Malaria diagnosis and parasite status [EUPATH_0000338]": malaria_diagnosis_parasite,
                "Malaria treatment [EUPATH_0000740]": malaria_treatment,
                "Muscle aches [EUPATH_0000252]": muscle_aches,
                "Muscle aches duration (days) [EUPATH_0000162]": muscle_aches_duration,
                "Non-malaria medication [EUPATH_0000059]": non_malaria_medication,
                "Other diagnosis [EUPATH_0000317]": other_diagnosis,
                "Other medical complaint [EUPATH_0020002]": other_medical_complaint,
                "Plasmodium gametocytes present, by microscopy [EUPATH_0000207]": plas_gam_present,
                "Seizures [SYMP_0000124]": seizures,
                "Seizures duration (days) [EUPATH_0000163]": seizures_duration,
                "Severe malaria criteria [EUPATH_0000046]": severe_malaria_criteria,
                "Subjective fever [EUPATH_0000100]": subjective_fever,
                "Submicroscopic Plasmodium present, by LAMP [EUPATH_0000487]": submic_plas_present,
                "Temperature (C) [EUPATH_0000110]": round(current_temperature, 1),
                "Visit date [EUPATH_0000091]": current_date,
                "Visit type [EUPATH_0000311]": visit_type,
                "Vomiting [HP_0002013]": vomiting,
                "Vomiting duration (days) [EUPATH_0000165]": vomiting_duration,
                "Weight (kg) [EUPATH_0000732]": round(current_weight * 2) / 2,
                "real": 999
            }

            # append imputed row to patient
            imputed_patient = imputed_patient.append(imputed_row, ignore_index=True)
            day_difference -= 1
            current_age += 1/365

        observation_id += 1
        current_date += delta

# ================================================================================================

    # Round 3
    # imputation after last visit date

    observation_id += 1
    current_age = last_row[age_string] + 1/365
    current_date += delta

    abd_pain = anorexia = cough = diarrhoea = fatigue = febrile = headache =\
        jaundice = joints_pain = muscle_aches = seizures = vomiting = 'Unable to assess'

    abd_pain_duration = anorexia_duration = cough_duration = fatigue_duration =\
        febrile_duration = headache_duration = joints_pain_duration =\
        muscle_aches_duration = seizures_duration = vomiting_duration = 0

    days_since_enrollment = np.nan

    # set height as NAN for children and constant for adults (may change for children in the future)
    height = np.nan
    if same_height_throughout:
        height = first_row[height_string]

    while current_date <= series_end_date:

        imputed_row = {
            "Observation_Id": observation_id, 
            "Participant_Id": patient_id,
            "Household_Id": household_id,
            "Abdominal pain [HP_0002027]": abd_pain,
            "Abdominal pain duration (days) [EUPATH_0000154]": abd_pain_duration,
            "Admitting hospital [EUPATH_0000318]": admitting_hospital,
            "Age at visit (years) [EUPATH_0000113]": round(current_age, 2),
            "Anorexia [SYMP_0000523]": anorexia,
            "Anorexia duration (days) [EUPATH_0000155]": anorexia_duration,
            "Asexual Plasmodium parasite density, by microscopy [EUPATH_0000092]": ax_plas_density,
            "Asexual Plasmodium parasites present, by microscopy [EUPATH_0000048]": ax_plas_present,
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
            "Height (cm) [EUPATH_0010075]": str(int(height)),
            "Hemoglobin (g/dL) [EUPATH_0000047]": average_haemoglobin,
            "Hospital admission date [EUPATH_0000319]": hospital_admission_date,
            "Hospital discharge date [EUPATH_0000320]": hospital_discharge_date,
            "ITN last night [EUPATH_0000216]": itn,
            "Jaundice [HP_0000952]": jaundice,
            "Jaundice duration (days) [EUPATH_0000160]": jaundice_duration,
            "Joint pains [SYMP_0000064]": joint_pains,
            "Joint pains duration (days) [EUPATH_0000161]": joint_pains_duration,
            "Malaria diagnosis [EUPATH_0000090]": malaria_diagnosis,
            "Malaria diagnosis and parasite status [EUPATH_0000338]": malaria_diagnosis_parasite,
            "Malaria treatment [EUPATH_0000740]": malaria_treatment,
            "Muscle aches [EUPATH_0000252]": muscle_aches,
            "Muscle aches duration (days) [EUPATH_0000162]": muscle_aches_duration,
            "Non-malaria medication [EUPATH_0000059]": non_malaria_medication,
            "Other diagnosis [EUPATH_0000317]": other_diagnosis,
            "Other medical complaint [EUPATH_0020002]": other_medical_complaint,
            "Plasmodium gametocytes present, by microscopy [EUPATH_0000207]": plas_gam_present,
            "Seizures [SYMP_0000124]": seizures,
            "Seizures duration (days) [EUPATH_0000163]": seizures_duration,
            "Severe malaria criteria [EUPATH_0000046]": severe_malaria_criteria,
            "Subjective fever [EUPATH_0000100]": subjective_fever,
            "Submicroscopic Plasmodium present, by LAMP [EUPATH_0000487]": submic_plas_present,
            "Temperature (C) [EUPATH_0000110]": average_temperature,
            "Visit date [EUPATH_0000091]": current_date,
            "Visit type [EUPATH_0000311]": visit_type,
            "Vomiting [HP_0002013]": vomiting,
            "Vomiting duration (days) [EUPATH_0000165]": vomiting_duration,
            "Weight (kg) [EUPATH_0000732]": average_weight,
            "real": 999
        }

        # append imputed row to patient
        imputed_patient = imputed_patient.append(imputed_row, ignore_index=True)
        current_age += 1/365
        observation_id += 1
        current_date += delta

    break

cols = list(imputed_patient.columns)
cols = [cols[-1]] + cols[:-1]
imputed_patient = imputed_patient[cols]

# display results of first two rounds so far
pd.set_option('display.max_rows', None)
diagnosis = imputed_patient[["Observation_Id", height_string, visit_date_string, 'real']]
#print(diagnosis.sort_values([visit_date_string]))

imputed_patient = imputed_patient.sort_values([visit_date_string])
imputed_patient.to_csv('first_patient_imputed.tsv', sep='\t', index=False)

print("--- %s seconds ---" % (time.time() - start_time))

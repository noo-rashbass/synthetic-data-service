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

columns = [
    'real', 'Observation_Id', 'Participant_Id', 'Household_Id', 'Abdominal pain [HP_0002027]',
	'Abdominal pain duration (days) [EUPATH_0000154]', 'Admitting hospital [EUPATH_0000318]',
    age_string, 'Anorexia [SYMP_0000523]', 'Anorexia duration (days) [EUPATH_0000155]',
	'Asexual Plasmodium parasite density, by microscopy [EUPATH_0000092]',
	'Asexual Plasmodium parasites present, by microscopy [EUPATH_0000048]',
	'Basis of complicated diagnosis [EUPATH_0000316]', 'Complicated malaria [EUPATH_0000040]',
	'Cough [SYMP_0000614]', 'Cough duration (days) [EUPATH_0000156]',
    'Days since enrollment [EUPATH_0000191]', 'Diagnosis at hospitalization [EUPATH_0000638]',
    'Diarrhea [DOID_13250]', 'Diarrhea duration (days) [EUPATH_0000157]',
    'Fatigue [SYMP_0019177]', 'Fatigue duration (days) [EUPATH_0000158]',
    'Febrile [EUPATH_0000097]', 'Fever, subjective duration (days) [EUPATH_0000164]',
    'Headache [HP_0002315]', 'Headache duration (days) [EUPATH_0000159]',
    height_string, haemoglobin_string,
	'Hospital admission date [EUPATH_0000319]', 'Hospital discharge date [EUPATH_0000320]',
	'ITN last night [EUPATH_0000216]',
    'Jaundice [HP_0000952]', 'Jaundice duration (days) [EUPATH_0000160]',
    'Joint pains [SYMP_0000064]', 'Joint pains duration (days) [EUPATH_0000161]',
    'Malaria diagnosis [EUPATH_0000090]',
    'Malaria diagnosis and parasite status [EUPATH_0000338]',
    'Malaria treatment [EUPATH_0000740]',
    'Muscle aches [EUPATH_0000252]', 'Muscle aches duration (days) [EUPATH_0000162]',
    'Non-malaria medication [EUPATH_0000059]',
	'Other diagnosis [EUPATH_0000317]', 'Other medical complaint [EUPATH_0020002]',
	'Plasmodium gametocytes present, by microscopy [EUPATH_0000207]', 'Seizures [SYMP_0000124]',
	'Seizures duration (days) [EUPATH_0000163]', 'Severe malaria criteria [EUPATH_0000046]',
	'Subjective fever [EUPATH_0000100]',
	'Submicroscopic Plasmodium present, by LAMP [EUPATH_0000487]',
	temperature_string, visit_date_string, 'Visit type [EUPATH_0000311]', 'Vomiting [HP_0002013]',
	'Vomiting duration (days) [EUPATH_0000165]', weight_string
]

data[visit_date_string] = pd.to_datetime(data[visit_date_string])

data = data.assign(real=1000)
imputed_list = []
imputed_rows = np.array([])

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

        imputed_row = [
            imputed_obs_id, patient_id, household_id, abd_pain, abd_pain_duration,
            admitting_hospital, round(current_age, 2), anorexia, anorexia_duration,
            ax_plas_density, ax_plas_present, complex_diagnosis_basis, complicated_malaria,
            cough, cough_duration, days_since_enrollment, diagnosis_at_hospital, diarrhoea,
            diarrhoea_duration, fatigue, fatigue_duration, febrile, febrile_duration, headache,
            headache_duration, str(int(height)), average_haemoglobin, hospital_admission_date,
            hospital_discharge_date, itn, jaundice, jaundice_duration, joint_pains,
            joint_pains_duration, malaria_diagnosis, malaria_diagnosis_parasite,
            malaria_treatment, muscle_aches, muscle_aches_duration, non_malaria_medication,
            other_diagnosis, other_medical_complaint, plas_gam_present, seizures,
            seizures_duration, severe_malaria_criteria, subjective_fever, submic_plas_present,
            average_temperature, current_date, visit_type, vomiting, vomiting_duration,
            weight, 999
        ]

        # append imputed row to patient
        imputed_list.append(imputed_row)
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

            imputed_row = [
                imputed_obs_id, patient_id, household_id, abd_pain, abd_pain_duration,
                admitting_hospital, round(current_age, 2), anorexia, anorexia_duration,
                ax_plas_density, ax_plas_present, complex_diagnosis_basis, complicated_malaria,
                cough, cough_duration, days_since_enrollment, diagnosis_at_hospital, diarrhoea,
                diarrhoea_duration, fatigue, fatigue_duration, febrile, febrile_duration,
                headache, headache_duration, str(int(current_height)),
                round(current_haemoglobin, 1), hospital_admission_date, hospital_discharge_date,
                itn, jaundice, jaundice_duration, joint_pains, joint_pains_duration,
                malaria_diagnosis, malaria_diagnosis_parasite, malaria_treatment, muscle_aches,
                muscle_aches_duration, non_malaria_medication, other_diagnosis,
                other_medical_complaint, plas_gam_present, seizures, seizures_duration,
                severe_malaria_criteria, subjective_fever, submic_plas_present,
                round(current_temperature, 1), current_date, visit_type, vomiting,
                vomiting_duration, round(current_weight * 2) / 2, 999
            ]

            # append imputed row to patient
            imputed_list.append(imputed_row)
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

        imputed_row = [
            imputed_obs_id, patient_id, household_id, abd_pain, abd_pain_duration,
            admitting_hospital, round(current_age, 2), anorexia, anorexia_duration,
            ax_plas_density, ax_plas_present, complex_diagnosis_basis, complicated_malaria,
            cough, cough_duration, days_since_enrollment, diagnosis_at_hospital, diarrhoea,
            diarrhoea_duration, fatigue, fatigue_duration, febrile, febrile_duration, headache,
            headache_duration, str(int(height)), average_haemoglobin, hospital_admission_date,
            hospital_discharge_date, itn, jaundice, jaundice_duration, joint_pains,
            joint_pains_duration, malaria_diagnosis, malaria_diagnosis_parasite,
            malaria_treatment, muscle_aches, muscle_aches_duration, non_malaria_medication,
            other_diagnosis, other_medical_complaint, plas_gam_present, seizures,
            seizures_duration, severe_malaria_criteria, subjective_fever, submic_plas_present,
            average_temperature, current_date, visit_type, vomiting, vomiting_duration,
            average_weight, 999
        ]

        # append imputed row to patient
        imputed_list.append(imputed_row)
        current_age += 1/365
        observation_id += 1
        current_date += delta

    break

imputed_rows = np.array(imputed_list)
print(imputed_rows)
imputed_data = pd.DataFrame(imputed_rows, columns)
data.append(imputed_data, ignore_index=True)

cols = list(data.columns)
cols = [cols[-1]] + cols[:-1]
data = data[cols]

# display results of first two rounds so far
#pd.set_option('display.max_rows', None)
#diagnosis = imputed_patient[["Observation_Id", height_string, visit_date_string, 'real']]
#print(diagnosis.sort_values([visit_date_string]))

imputed_patient = imputed_patient.sort_values([visit_date_string])
imputed_patient.to_csv('new_imputed.tsv', sep='\t', index=False)

print("--- %s seconds ---" % (time.time() - start_time))

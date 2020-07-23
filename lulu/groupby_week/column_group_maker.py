import pandas as pd

df = pd.read_csv('column_grouping_punchcard.csv').fillna(0)
# print(df.head())

group_names = df.columns.tolist()
column_names = ['Observation_Id', 'Participant_Id', 'Household_Id', 'Abdominal pain [HP_0002027]', 'Abdominal pain duration (days) [EUPATH_0000154]', 'Admitting hospital [EUPATH_0000318]', 'Age at visit (years) [EUPATH_0000113]', 'Anorexia [SYMP_0000523]', 'Anorexia duration (days) [EUPATH_0000155]', 'Asexual Plasmodium parasite density, by microscopy [EUPATH_0000092]', 'Asexual Plasmodium parasites present, by microscopy [EUPATH_0000048]', 'Basis of complicated diagnosis [EUPATH_0000316]', 'Complicated malaria [EUPATH_0000040]', 'Cough [SYMP_0000614]', 'Cough duration (days) [EUPATH_0000156]', 'Days since enrollment [EUPATH_0000191]', 'Diagnosis at hospitalization [EUPATH_0000638]', 'Diarrhea [DOID_13250]', 'Diarrhea duration (days) [EUPATH_0000157]', 'Fatigue [SYMP_0019177]', 'Fatigue duration (days) [EUPATH_0000158]', 'Febrile [EUPATH_0000097]', 'Fever, subjective duration (days) [EUPATH_0000164]', 'Headache [HP_0002315]', 'Headache duration (days) [EUPATH_0000159]', 'Height (cm) [EUPATH_0010075]', 'Hemoglobin (g/dL) [EUPATH_0000047]', 'Hospital admission date [EUPATH_0000319]', 'Hospital discharge date [EUPATH_0000320]', 'ITN last night [EUPATH_0000216]', 'Jaundice [HP_0000952]', 'Jaundice duration (days) [EUPATH_0000160]', 'Joint pains [SYMP_0000064]', 'Joint pains duration (days) [EUPATH_0000161]', 'Malaria diagnosis [EUPATH_0000090]', 'Malaria diagnosis and parasite status [EUPATH_0000338]', 'Malaria treatment [EUPATH_0000740]', 'Muscle aches [EUPATH_0000252]', 'Muscle aches duration (days) [EUPATH_0000162]', 'Non-malaria medication [EUPATH_0000059]', 'Other diagnosis [EUPATH_0000317]', 'Other medical complaint [EUPATH_0020002]', 'Plasmodium gametocytes present, by microscopy [EUPATH_0000207]', 'Seizures [SYMP_0000124]', 'Seizures duration (days) [EUPATH_0000163]', 'Severe malaria criteria [EUPATH_0000046]', 'Subjective fever [EUPATH_0000100]', 'Submicroscopic Plasmodium present, by LAMP [EUPATH_0000487]', 'Temperature (C) [EUPATH_0000110]', 'Visit date [EUPATH_0000091]', 'Visit type [EUPATH_0000311]', 'Vomiting [HP_0002013]', 'Vomiting duration (days) [EUPATH_0000165]', 'Weight (kg) [EUPATH_0000732]']
print('Punchcard includes these group names:')
print(group_names)
# print(df.to_numpy())

def make_group_list(df, group_name):
	punchcard = df[group_name]
	group = [column_names[i] for i in range(len(punchcard)) if punchcard[i] == 1]
	return group

# YesNo = make_group_list('YesNo')

def make_groups_dict(df, group_names, column_names):
	return {name:make_group_list(df, name) for name in group_names}


groups_dict = make_groups_dict(df, group_names, column_names)
# print(groups_dict['YesNo'])
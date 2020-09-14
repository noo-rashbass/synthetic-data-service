# thsi file contains functions for cleaning prism data and transforming the data 
# into a format acceptable by the doppelGANger

import pandas as pd 
import numpy as np

def clean_prism(data, first_dday_as_attr):
    """accepts original prism dataframe and returns a cleaned version
    **NOTE: that this function is highly specifc to the prism dataset 
    and the data preparation done are based on what we think is suitable.
    For other datasets, you should have your own way for data preparation"""

    # select columns that are wanted
    data2 = data[['Participant_Id','Visit date [EUPATH_0000091]', 
    "Abdominal pain duration (days) [EUPATH_0000154]","Age at visit (years) [EUPATH_0000113]", 
    "Anorexia duration (days) [EUPATH_0000155]", "Asexual Plasmodium parasite density, by microscopy [EUPATH_0000092]", 
    "Cough duration (days) [EUPATH_0000156]", "Diarrhea duration (days) [EUPATH_0000157]", 
    "Fatigue duration (days) [EUPATH_0000158]", "Fever, subjective duration (days) [EUPATH_0000164]", 
    "Headache duration (days) [EUPATH_0000159]", "Height (cm) [EUPATH_0010075]", "Hemoglobin (g/dL) [EUPATH_0000047]", 
    "Joint pains duration (days) [EUPATH_0000161]", "Muscle aches duration (days) [EUPATH_0000162]", 
    "Temperature (C) [EUPATH_0000110]", "Vomiting duration (days) [EUPATH_0000165]", "Weight (kg) [EUPATH_0000732]", 
    'Complicated malaria [EUPATH_0000040]', "Febrile [EUPATH_0000097]", "ITN last night [EUPATH_0000216]", 
    "Malaria diagnosis [EUPATH_0000090]", "Malaria diagnosis and parasite status [EUPATH_0000338]", 
    "Malaria treatment [EUPATH_0000740]", "Plasmodium gametocytes present, by microscopy [EUPATH_0000207]", 
    "Submicroscopic Plasmodium present, by LAMP [EUPATH_0000487]", 
    "Visit type [EUPATH_0000311]"]].copy().sort_values(by =['Participant_Id', 'Visit date [EUPATH_0000091]']).reset_index(drop=True)

    # rename columns
    data2.columns = ['id', 'date', 'ab_pain_dur', 'age', 'aneroxia_dur', 'plasmodium_density', 
    'cough_dur', 'diarrhea_dur', 'fatigue_dur', 'fever_dur', 'headache_dur', 
    'height', 'hemoglobin', 'joint_pain_dur', 'muscle_ache_dur', 'temp', 'vomit_dur', 
    'weight', 'complicated_malaria','febrile', 'ITN', 'malaria', 'malaria_parasite', 
    'malaria_treatment', 'plasmodium_gametocytes', 'plasmodium_lamp', 'visit_type']

    # drop a row which is mostly NAs
    data2 = data2.drop(44432)

    # fill NAs in duration columns and plasmodium density column with 0
    dur_cols = ['ab_pain_dur', 'aneroxia_dur', 'plasmodium_density', 'cough_dur', 
        'diarrhea_dur', 'fatigue_dur', 'fever_dur', 'headache_dur', 'joint_pain_dur', 'muscle_ache_dur', 'vomit_dur']
    for col in dur_cols:
        data2[col] = data2[col].fillna(0)

    # fill NAs in numerical columns by interpolation
    num_cols = ['height', 'hemoglobin', 'temp', 'weight']
    for col in num_cols:
        data2[col] = data2[col].interpolate(method='linear')

    # fill NAs in categorical columns with new category "not applicable"/"no result" etc
    data2['plasmodium_lamp'] = data2['plasmodium_lamp'].fillna('no_result')
    data2['ITN'] = data2['ITN'].fillna('not applicable')
    data2['complicated_malaria'] = data2['complicated_malaria'].fillna('not_assessed')
    data2['plasmodium_gametocytes'] = data2['plasmodium_gametocytes'].fillna('No')

    # replace white spaces with underscores so that it won't create trouble later
    data2 = data2.replace(' ', '_', regex=True)

    # convert categorical column values to lowercase
    # one hot encode categorical columns
    cat_cols = ['complicated_malaria', 'febrile', 'ITN', 'malaria', 'malaria_parasite', 
    'malaria_treatment', 'plasmodium_gametocytes', 'plasmodium_lamp', 'visit_type']
    for col in cat_cols:
        data2[col] = data2[col].map(lambda x: x.lower() if isinstance(x,str) else x)
        one_hot_cols = pd.get_dummies(data2[col], prefix=col)
        data2 = pd.concat([data2, one_hot_cols], axis=1)

    if first_dday_as_attr:
        # find delta day between visits
        # for first visit, we fill in with 0 for now
        data2['id_diff'] = data2['id'].diff()
        data2['date'] = pd.to_datetime(data2['date'])
        data2['dday'] = data2['date'].diff()
        def fill_first_dday(row):
            if row['id_diff'] != 0:
                return (row['date']- row['date']) # to get 0 in datetime format
            else:
                return row['dday']
        data2['dday'] = data2.apply(fill_first_dday, axis=1)
        data2['dday'] = data2['dday'].dt.days.astype('int16') # get int value from datetime format

        # get delta day for first visit
        # this is a separate column from above as we will treat it as an attribute rather than feature
        earliest_date = min(data2['date'])
        def get_first_dday(row, earliest_date):
            if row['id_diff'] != 0:
                return (row['date']- earliest_date)
        data2['first_dday'] = data2.apply(get_first_dday,args=(earliest_date,), axis=1)
        data2['first_dday'] = data2['first_dday'].fillna(method='ffill')
        data2['first_dday'] = data2['first_dday'].dt.days.astype('int16')

        # only take patiente with more than 5 visits
        data_5above = data2[(data2.groupby('id')['id'].transform('size') >= 5)].reset_index(drop=True)
        data_5above = data_5above.sort_values(by=['id', 'date'])

        # remove unwanted columns
        data_5above = data_5above.drop(columns=['date','complicated_malaria', 'febrile', 'ITN', 
        'malaria', 'malaria_parasite', 'malaria_treatment', 'plasmodium_gametocytes', 'plasmodium_lamp', 
        'visit_type', 'id_diff'])

        # write data into a csv
        # data_5above.to_csv('data/ori_prism_cleaned.csv', index=False)
    
    else:
        # find delta day between visits
        # for first visit, we find first visit date of patient - first ever visit in dataset
        data2['id_diff'] = data2['id'].diff()
        data2['date'] = pd.to_datetime(data2['date'])
        data2['dday'] = data2['date'].diff()

        #first ever visit in dataset
        earliest_date = min(data2['date'])

        def get_first_dday(row, earliest_date):
            if row['id_diff'] != 0:
                return (row['date'] - earliest_date)
            else:
                return row['dday']
        data2['dday'] = data2.apply(get_first_dday, args=(earliest_date,), axis=1)
        data2['dday'] = data2['dday'].dt.days.astype('int16') # get int value from datetime format

        # only take patiente with more than 5 visits
        data_5above = data2[(data2.groupby('id')['id'].transform('size') >= 5)].reset_index(drop=True)
        data_5above = data_5above.sort_values(by=['id', 'date'])

        # remove unwanted columns
        data_5above = data_5above.drop(columns=['date','complicated_malaria', 'febrile', 'ITN', 'malaria', 
        'malaria_parasite', 'malaria_treatment', 'plasmodium_gametocytes', 'plasmodium_lamp', 'visit_type', 'id_diff'])

        # write data into a csv
        # data_5above.to_csv('data_attr/ori_prism_cleaned.csv', index=False)

    return data_5above


def real_data_loading(path):
    """
    takes in cleaned dataset 
    and returns normalised (between 0 to 1) features, gen_flag, attributes
    to be saved in data_train.npz, 
    min_ and max_ for renormalization,
    feature_col and attribute_col to name output file columns
    """

    data = pd.read_csv(path)
    # fill in any NAs that have been missed out
    data.interpolate(method = 'linear', inplace=True)

    # normalizing data
    min_val = data.min()
    max_val = data.max()
    data = (data - min_val) / (max_val - min_val + 1e-7)
    id_unique = data.id.unique()

    # CHANGE feature cols according to dataset
    feature_cols = ['ab_pain_dur', 'age', 'aneroxia_dur', 'plasmodium_density', 'cough_dur', 'diarrhea_dur', 
        'fatigue_dur', 'fever_dur', 'headache_dur', 'height', 'hemoglobin', 'joint_pain_dur', 'muscle_ache_dur', 
        'temp', 'vomit_dur', 'weight', 'complicated_malaria_no', 'complicated_malaria_not_assessed', 
        'complicated_malaria_yes', 'febrile_no', 'febrile_yes', 'ITN_no', 'ITN_not_applicable', 'ITN_yes',
        'malaria_no', 'malaria_yes',
        'malaria_parasite_blood_smear_indicated_but_not_done',
        'malaria_parasite_blood_smear_negative_/_lamp_negative',
        'malaria_parasite_blood_smear_negative_/_lamp_not_done',
        'malaria_parasite_blood_smear_negative_/_lamp_positive',
        'malaria_parasite_blood_smear_not_indicated',
        'malaria_parasite_blood_smear_positive_/_no_malaria',
        'malaria_parasite_symptomatic_malaria',
        'malaria_treatment_artmether-lumefantrine_for_uncomplicated_malaria',
        'malaria_treatment_no_malaria_medications_given',
        'malaria_treatment_quinine_for_uncomplicated_malaria_in_the_1st_trimester_of_pregnancy',
        'malaria_treatment_quinine_for_uncomplicated_malaria_within_14_days_of_a_previous_treatment_for_malaria',
        'malaria_treatment_quinine_or_artesunate_for_complicated_malaria',
        'plasmodium_gametocytes_no', 'plasmodium_gametocytes_yes',
        'plasmodium_lamp_negative', 'plasmodium_lamp_no_result',
        'plasmodium_lamp_positive', 'visit_type_enrollment',
        'visit_type_scheduled_visit', 'visit_type_unscheduled_visit', 'dday']

    # CHANGE attribute columns according to dataset
    attribute_cols = ['first_dday']

    features =[] 
    attributes = []
    gen_flag = []

    # iterate over each id
    for i in id_unique:
        #get features for each participant
        child = np.array(data.loc[data['id'] == i][feature_cols])

        if len(child) >= 5:
            # get padded gen_flag according to length of time series for each child
            gen_flag.append(np.concatenate([np.ones(len(child)), np.zeros(130-len(child))]))   
            # get padded features for each child
            child = np.pad(child, ((0, 130-len(child)), (0,0)))
            features.append(child)
            # get attributes for each child
            # CHANGE attr col according to dataset
            attributes.append(np.array(data.loc[data['id'] == i].iloc[0, -1])) #-1 for first_dday
            # attributes.append(np.squeeze(np.array(data.loc[data['id'] == i][attribute_cols])[0]))

    min_val = min_val.drop('id')
    max_val = max_val.drop('id')
  
    return np.array(features), np.array(attributes), np.array(gen_flag), min_val, max_val, feature_cols, attribute_cols
# this file contains function to convert generated data back to the original
# format the original data is in

import numpy as np
import pandas as pd

# currently works only for one feature, where first dday is the feature
def gen_data_loading(path, feature_cols, attribute_cols, min_val, max_val, first_dday_as_attr, seq_len=130):
    """
    loads in generated npz file and returns an intermediate csv for generated data (suitable for evaluation purposes)
    Args:
    path: path to npz file from doppelganger
    feature_cols: column names for features in dataset
    attribute_cols: column names for attributes in dataset
    min_val: min val for each of the normalized columns for renormalisation
    max_val: max val for each of the normalized columns for renormalisation
    seq_len: max sequence length for each patient
    """

    data = np.load(path)
    
    # get generate features (temporal data)
    data_feat = np.clip(data['data_feature'], 0, 1)
    dim = data['data_feature'].shape[2]
    data_stack = data_feat.reshape((-1,dim))

    # create a dataframe from the array
    data_df = pd.DataFrame(data_stack)
    data_df.columns = feature_cols

    # get generated attributes (static data)
    if first_dday_as_attr:
        data_attr = data['data_attribute']
        data_attr = np.repeat(data_attr, seq_len)
        data_df['first_dday'] = data_attr # we only have one column for attribute

    data_df['id'] = data_df.index // seq_len + 1

    # remove padded columns
    data_df = data_df.drop(data_df[(data_df.weight == 0) & (data_df.height == 0)].index)
    
    #renormalization
    cols_to_renormalize = feature_cols
    if first_dday_as_attr:
        cols_to_renormalize = feature_cols + attribute_cols
    data_df[cols_to_renormalize] = data_df[cols_to_renormalize] * (np.array(max_val) - np.array(min_val)) + np.array(min_val)

    ## converts probabilities into hard 0s and 1s
    # list of categorical columns
    cat_cols = ['complicated_malaria', 'febrile', 'ITN', 'malaria_parasite', 'malaria_treatment', 'plasmodium_gametocytes', 'plasmodium_lamp', 'visit_type']
    
    # for each categorical data
    for cat_col in cat_cols:
        # get the one hot encoded columns for that category
        related_cols = [col for col in data_df if col.startswith(cat_col)]
        # find the max probability for each row
        max_related_cols = data_df[related_cols].max(axis=1)
        # if value = max in row, then 1, all other values will be 0
        for col in related_cols:
            data_df[col] = np.where(data_df[col] == max_related_cols, 1, 0)
    
    # malaria is not included in loop above bcoz columns that start with 'malaria_'
    # includes columns that start with 'malaria_parasite', 'malaria_treatment'
    # which messes things up
    data_df[['malaria_no', 'malaria_yes']] = data_df[['malaria_no', 'malaria_yes']].round(0)

    # reorder columns so that id becomes first column as in original csv
    cols = data_df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    data_df = data_df[cols]

    # write output to csv file
    # data_df.to_csv('data_attr/gen_prism_int_e200.csv', index=False)

    return data_df
    

def convert_df_to_ori_format(dataframe, first_dday_as_attr):
  
    """
    converts df that contains one hot encoding, unrounded values and dday values
    back to the original format where the data is received, with categorical values 
    in the same column and date in which an event occured
    
    Args:
    dataframe: dataframe with one-hot encoding, dday
    
    Returns:
    df: dataframe with one-hot encoding reversed, dday converted to date
    """

    df = dataframe.copy()

    cat_cols = ['complicated_malaria', 'febrile', 'ITN', 'malaria_parasite', 'malaria_treatment', 'plasmodium_gametocytes', 'plasmodium_lamp', 'visit_type']
    
    # for each categorical data
    for cat_col in cat_cols:
        # get the one hot encoded columns for that category
        related_cols = [col for col in df if col.startswith(cat_col)]
        # "reverse" the one hot encoding
        df[cat_col] = df[related_cols].idxmax(1).str.replace(cat_col+'_', '')
        # remove extra columns
        df = df.drop(columns=related_cols)

    # malaria is not included in loop above bcoz columns that start with 'malaria_'
    # includes columns that start with 'malaria_parasite', 'malaria_treatment'
    # which messes things up
    df['malaria'] = df[['malaria_yes', 'malaria_no']].idxmax(1).str.replace('malaria_', '')
    df = df.drop(columns=['malaria_yes', 'malaria_no'])

    # replace underscores with white spaces
    # revert back to original format of data
    df = df.replace('_', ' ', regex=True)

    # round off data according to original decimal places
    dur_cols = [col for col in df if col.endswith('_dur')]
    df[dur_cols] = df[dur_cols].round()
    df[['dday']] = df[['dday']].round()
    if first_dday_as_attr:
        df[['first_dday']] = df[['first_dday']].round()
    df[['age', 'hemoglobin']] = df[['age', 'hemoglobin']].round(2)
    df[['height', 'temp', 'weight', 'plasmodium_density']] = df[['height', 'temp', 'weight','plasmodium_density']].round(1)

    # convert dday back to date
    # first_dday_as_attr = True
    # first_dday: first visit day for that patient (to capture global distribution)
    # dday: diff in days between subsequent visit
    # first_dday_as_attr = False
    # dday: first value for each id= first_dday explained above, the rest of the dday= dday as explained above
    earliest_date = pd.to_datetime('2011-07-29')
    df['day_num'] = df.groupby('id')['dday'].cumsum()
    if first_dday_as_attr:
        df['date'] = pd.to_timedelta(df['first_dday'], unit='D') + earliest_date
        df['date'] = df['date'] + pd.to_timedelta(df['day_num'], unit='D')
    else:
        df['day_num'] = df.groupby('id')['dday'].cumsum()
        df['date'] = pd.to_timedelta(df['day_num'], unit='D') + earliest_date

    # select and reorder wanted columns
    rearranged_cols = ['id', 'date', 'ab_pain_dur', 'age', 'aneroxia_dur', 'plasmodium_density', 'cough_dur', 'diarrhea_dur', 
            'fatigue_dur', 'fever_dur', 'headache_dur', 'height', 'hemoglobin', 'joint_pain_dur', 'muscle_ache_dur', 
            'temp', 'vomit_dur', 'weight'] + cat_cols + ['malaria']
    df = df[rearranged_cols]

    # rename columns to original column names
    df.columns = ['Participant_Id','Visit date [EUPATH_0000091]', 
        "Abdominal pain duration (days) [EUPATH_0000154]","Age at visit (years) [EUPATH_0000113]", 
        "Anorexia duration (days) [EUPATH_0000155]", "Asexual Plasmodium parasite density, by microscopy [EUPATH_0000092]", 
        "Cough duration (days) [EUPATH_0000156]", "Diarrhea duration (days) [EUPATH_0000157]", 
        "Fatigue duration (days) [EUPATH_0000158]", "Fever, subjective duration (days) [EUPATH_0000164]", 
        "Headache duration (days) [EUPATH_0000159]", "Height (cm) [EUPATH_0010075]", "Hemoglobin (g/dL) [EUPATH_0000047]", 
        "Joint pains duration (days) [EUPATH_0000161]", "Muscle aches duration (days) [EUPATH_0000162]", 
        "Temperature (C) [EUPATH_0000110]", "Vomiting duration (days) [EUPATH_0000165]", "Weight (kg) [EUPATH_0000732]", 
        'Complicated malaria [EUPATH_0000040]', "Febrile [EUPATH_0000097]", "ITN last night [EUPATH_0000216]", 
        "Malaria diagnosis and parasite status [EUPATH_0000338]", 
        "Malaria treatment [EUPATH_0000740]", "Plasmodium gametocytes present, by microscopy [EUPATH_0000207]", 
        "Submicroscopic Plasmodium present, by LAMP [EUPATH_0000487]", 
        "Visit type [EUPATH_0000311]", "Malaria diagnosis [EUPATH_0000090]"]

    return df

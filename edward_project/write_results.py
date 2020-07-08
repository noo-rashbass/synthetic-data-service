#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file can be imported as a module and contains the following functions:
    * get_totals_from_db - Reads group counts from a table in an SQL database into a pandas DataFrame.
    * write_counts_to_csv - Writes group counts from a table in an SQL database to a .csv file.
"""


import pandas as pd

import queries
from populations import pop_queries
from params import filepath_dictionary, field_list_dict


__author__ = 'Edward Pearce'
__copyright__ = 'Copyright 2019, Simulacrum Test Suite'
__credits__ = ['Edward Pearce']
__license__ = 'MIT'
__version__ = '1.0.0'
__maintainer__ = 'Edward Pearce'
__email__ = 'edward.pearce@phe.gov.uk'
__status__ = 'Development'


def get_totals_from_db(count_type, key, db):
    r"""Reads group counts from a table in an SQL database into a pandas DataFrame.
    
    It is necessary to have access to the SQL database (via an `sqlalchemy.engine` object) to use this function.
    
    Parameters
    ----------
    count_type : str
        The list of fields or pairs of fields which we group by for counting is indicated by the `count_type`variable.
    key : str
        The table in the SQL database for which we calculate the group counts is indicated by the `key` variable.
    db: An instance of an `sqlalchemy.engine`
        This is the connection to your database management system.

    Returns
    -------
    pandas DataFrame
        Returns the table of group counts as a pandas DataFrame
    """
    if count_type in ['univariate_categorical', 'univariate_dates']:
        num_variates = 1
    elif count_type in ['bivariate_categorical', 'categorical_cross_diagnosis_date', 'categorical_cross_surgery_date', 'surgery_date_cross_diagnosis_date']:
        num_variates = 2
    return pd.read_sql_query(queries.make_totals_query(pop_queries[key], key, field_list=field_list_dict[count_type], num_variates=num_variates), db)


def write_counts_to_csv(count_type, key, db):
    r"""Writes group counts from a table in an SQL database to a .csv file.
    
    Obtains a chosen set of group counts from an SQL table, cleans and sorts values, and writes the results to a .csv file.
    
    Parameters
    ----------
    count_type : str
        The type of fields or pairs of fields that we would like to group by for counting
    key : str
        A key indicating the table in the SQL database from which we calculate group counts
    
    Returns
    -------
    Boolean
        Returns True if the function was executed successfully
    
    """
    # Read the raw table of counts data
    print('Getting the data from {} - calculating {} counts in SQL...'.format(key, count_type))
    frame = get_totals_from_db(count_type, key, db)
    print('Totals pulled from database successfully! ({} rows, {} columns)'.format(frame.shape[0], frame.shape[1]))
    
    print('Setting data types and cleaning values...')
    if count_type in ['univariate_categorical', 'univariate_dates']:
        frame['column_name'] = frame['column_name'].astype('category')
        frame['counts_'+key] = frame['counts_'+key].astype('uint32')
        if count_type == 'univariate_categorical':
            frame['val'] = frame.apply(lambda row: int(row['val']) if row['column_name'] == 'AGE' else row['val'], axis=1)
        elif count_type == 'univariate_dates':
            frame['val'] = pd.to_datetime(frame['val'], infer_datetime_format=True, errors='coerce')
        
        print('Sorting values...')
        frame = pd.concat([frame.loc[frame.column_name == col_name].sort_values(by='val') for col_name in field_list_dict[count_type]])

    elif count_type in ['bivariate_categorical', 'categorical_cross_diagnosis_date', 'categorical_cross_surgery_date', 'surgery_date_cross_diagnosis_date']:
        frame[['column_name1', 'column_name2']] = frame[['column_name1', 'column_name2']].astype('category')
        frame['counts_'+key] = frame['counts_'+key].astype('uint32')
        
        if count_type == 'bivariate_categorical':
            # By design, 'AGE' should always be in column_name2 when it appears in a pair
            frame['val1'] = frame.apply(lambda row: int(row['val1']) if row['column_name1'] == 'AGE' else row['val1'], axis=1)
            frame['val2'] = frame.apply(lambda row: int(row['val2']) if row['column_name2'] == 'AGE' else row['val2'], axis=1)
        elif count_type in ['categorical_cross_diagnosis_date', 'categorical_cross_surgery_date']:
            # By design, date fields should always be in column_name2 when appearing in the pairs
            frame['val1'] = frame.apply(lambda row: int(row['val1']) if row['column_name1'] == 'AGE' else row['val1'], axis=1)
            frame['val2'] = pd.to_datetime(frame['val2'], infer_datetime_format=True, errors='coerce')
        elif count_type == 'surgery_date_cross_diagnosis_date':
            frame['val1'] = pd.to_datetime(frame['val1'], infer_datetime_format=True, errors='coerce')
            frame['val2'] = pd.to_datetime(frame['val2'], infer_datetime_format=True, errors='coerce')
        
        print('Sorting values...')
        frame = pd.concat([frame.loc[(frame.column_name1 == pair[0]) & (frame.column_name2 == pair[1])]
                                   .sort_values(by=['val1', 'val2']) for pair in field_list_dict[count_type]])
    print('Data cleaned and sorted!\n Saving the results...')
    frame.to_csv(filepath_dictionary[count_type][key], index=False)
    print('Saved successfully at {} ! Function complete!'.format(filepath_dictionary[count_type][key]))
    return True

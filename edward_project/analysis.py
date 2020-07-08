#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file can be imported as a module and contains the following functions:
    * read_counts - read the tables of group counts results for a given count type (e.g. 'univariate categorical')
    * combine counts - join the tables of group counts into a single table for a given count type
"""

# Third-party imports
import pandas as pd

# Local packages
from params import filepath_dictionary, key_list, comparison_pairs


__author__ = 'Edward Pearce'
__copyright__ = 'Copyright 2019, Simulacrum Test Suite'
__credits__ = ['Edward Pearce']
__license__ = 'MIT'
__version__ = '1.0.0'
__maintainer__ = 'Edward Pearce'
__email__ = 'edward.pearce@phe.gov.uk'
__status__ = 'Development'


def read_counts(count_type):
    r"""Read the tables of group counts results for a given count type (e.g. 'univariate categorical')
    
    Returns a dictionary of pandas DataFrame objects whose keys are table aliases and values are tables of group counts.
    """
    # Initialise the dictionary where we will store the tables of group counts data
    counts_tables = dict()
    # Get the filepaths where we will be reading the data from
    filepaths = filepath_dictionary[count_type]
    # Read the group counts data according to the count type we chose into a DataFrame for each source table
    for key, filepath in filepaths.items():
        if count_type == 'univariate_categorical':        
            counts_tables[key] = pd.read_csv(filepath, dtype={'column_name': 'category', 'counts_'+key: 'uint32'})
        elif count_type == 'univariate_dates': 
            counts_tables[key] = pd.read_csv(filepath, parse_dates=[1], infer_datetime_format=True,
                                             dtype={'column_name': 'category', 'counts_'+key: 'uint32'})
        elif count_type == 'bivariate_categorical': 
            counts_tables[key] = pd.read_csv(filepath,
                                             dtype={'column_name1': 'category', 'column_name2': 'category', 'counts_'+key: 'uint32'})
        elif count_type == 'categorical_cross_diagnosis_date': 
            counts_tables[key] = pd.read_csv(filepath, usecols=[0, 2, 3, 4],
                                             parse_dates={'DIAGNOSISDATEBEST': [2]}, infer_datetime_format=True,
                                             dtype={'column_name1': 'category', 'counts_'+key: 'uint32'})
        elif count_type == 'categorical_cross_surgery_date': 
            counts_tables[key] = pd.read_csv(filepath, usecols=[0, 2, 3, 4],
                                             parse_dates={'DATE_FIRST_SURGERY': [2]}, infer_datetime_format=True,
                                             dtype={'column_name1': 'category', 'counts_'+key: 'uint32'})
        elif count_type == 'surgery_date_cross_diagnosis_date': 
            counts_tables[key] = pd.read_csv(filepath, usecols=[2, 3, 4],
                                             parse_dates={'DATE_FIRST_SURGERY': [0], 'DIAGNOSISDATEBEST': [1]},
                                             infer_datetime_format=True,
                                             dtype={'counts_'+key: 'uint32'})
    return counts_tables


# Specifies the columns which we use to join the tables of counts
join_cols = {'univariate_categorical': ['column_name', 'val'], 
              'univariate_dates': ['column_name', 'val'],
              'bivariate_categorical': ['column_name1', 'column_name2', 'val1', 'val2'], 
              'categorical_cross_diagnosis_date': ['column_name1', 'val1', 'DIAGNOSISDATEBEST'], 
              'categorical_cross_surgery_date': ['column_name1', 'val1', 'DATE_FIRST_SURGERY'], 
              'surgery_date_cross_diagnosis_date': ['DATE_FIRST_SURGERY', 'DIAGNOSISDATEBEST']}


def combine_counts(count_type, counts_tables):
    r"""Join pairs of counts tables for comparison for a given count type.
    
    Returns a dictionary of pandas DataFrames whose keys are pairs of table aliases and values are joined tables of group counts data.
    Uses the module parameter `comparison_pairs` to decide which tables to join.
    """
    comparison_tables = dict()
    for pair in comparison_pairs:
        # First, join the counts tables in the pairs we want to compare
        comparison_tables[pair] = pd.merge(counts_tables[pair[1]], counts_tables[pair[0]], 
                                           on=join_cols[count_type], how='outer')
        # Fill in missing count values with 0
        count_cols = ['counts_'+name for name in pair]
        comparison_tables[pair][count_cols] = comparison_tables[pair][count_cols].fillna(0, axis=1).astype('uint32')
    return comparison_tables


def compute_z_test(pair, comparison_table):
    table = comparison_table.copy()
    table['proportion_'+pair[0]] = table['counts_'+pair[0]]/pop_sizes[pair[0]]
    table['proportion_'+pair[1]] = table['counts_'+pair[1]]/pop_sizes[pair[1]]
    table['p_diff'] = table['proportion_'+pair[0]] - table['proportion_'+pair[1]]
    table['p_ave'] = (table['counts_'+pair[0]] + table['counts_'+pair[1]]) / (pop_sizes[pair[0]] + pop_sizes[pair[1]])
    table['z_test'] = table['p_diff']/np.sqrt(table['p_ave'] * (1 - table['p_ave']) * ((1/pop_sizes[pair[0]]) + (1/pop_sizes[pair[1]])))
    return table


def compute_chi2_test(pair, comparison_table):
    table = comparison_table.copy()
    table['proportion_'+pair[0]] = table['counts_'+pair[0]]/pop_sizes[pair[0]]
    table['proportion_'+pair[1]] = table['counts_'+pair[1]]/pop_sizes[pair[1]]
    table['expected_count_'+pair[0]] = pop_sizes[pair[0]] * table['proportion_'+pair[1]]
    table['squared_error'] = np.square(table['counts_'+pair[0]] - table['expected_count_'+pair[0]])
    table['pearson_chi2_test'] = table['squared_error'] / table['expected_count_'+pair[0]].where(table['expected_count_'+pair[0]] != 0, other=0.5)

    grouped = table[['column_name', 'pearson_chi2_test']].groupby(by='column_name')
    results = pd.concat([grouped.size(), grouped['pearson_chi2_test'].sum(), grouped.size() - 1], axis=1)
    results.columns = ['num_cats', 'pearson_chi2_test', 'degrees_of_freedom']
    # Apply the Wilson–Hilferty transformation to the chi-squared test statistics to approximately normalize them
    results['normalized_score'] = (np.cbrt(results['pearson_chi2_test']/results['degrees_of_freedom'])
                                   - (1 - 2/(9 * results['degrees_of_freedom']))) / (2/(9 * results['degrees_of_freedom']))
    return results


def compute_cdf(pair, comparison_table):
    table = comparison_table.copy().sort_values(by='val')
    table['proportion_'+pair[0]] = table['counts_'+pair[0]]/pop_sizes[pair[0]]
    table['proportion_'+pair[1]] = table['counts_'+pair[1]]/pop_sizes[pair[1]]
    cumsum_results = table[['column_name']+['proportion_'+key for key in pair]].groupby(by='column_name').cumsum()
    cumsum_results.columns = ['cdf_'+key for key in pair]
    results = pd.concat([table, cumsum_results], axis=1)
    return results


def compute_ks_test(pair, comparison_table):
    cdf_table = compute_cdf(pair, comparison_table)
    cdf_table['cdf_diff'] = (cdf_table['cdf_'+pair[0]] - cdf_table['cdf_'+pair[1]]).abs()
    results = cdf_table[['column_name', 'cdf_diff']].groupby(by='column_name').max()
    results.columns = ['ks_test_statistic']
    results['ks_scaled'] = results['ks_test_statistic'] * np.sqrt((pop_sizes[pair[0]] * pop_sizes[pair[1]])/(pop_sizes[pair[0]] + pop_sizes[pair[1]]))
    results['p_value'] = np.exp(-2 * np.square(results['ks_scaled']))
    return results


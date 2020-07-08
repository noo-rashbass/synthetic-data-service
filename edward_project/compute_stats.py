#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file can be imported as a module and contains the following functions:
    * compute_proportions - Compute category sizes as a proportion of the total population
    * compute_z_test - Compute z-test statistics based on value counts data from two populations
    * compute_chi2_test - Compute Pearson's chi-squared test statistics based on value counts data from two populations
    * compute_cdf - Compute cumulative distribution functions (CDF) based on ordered value counts data for a pair of populations
    * compute_ks_test - Compute Kolmogorov-Smirnov test statistics based on ordered value counts data by field from two populations
This module also contains the parameter `pop_sizes` which is a dictionary containing the number of data entries (rows) in the source cohort tables.
"""

# Third-party imports
import numpy as np
import pandas as pd


__author__ = 'Edward Pearce'
__copyright__ = 'Copyright 2019, Simulacrum Test Suite'
__credits__ = ['Edward Pearce']
__license__ = 'MIT'
__version__ = '1.0.0'
__maintainer__ = 'Edward Pearce'
__email__ = 'edward.pearce@phe.gov.uk'
__status__ = 'Development'


# Population sizes: The number of data entries (rows) in the source cohort tables. Used to calculate proportions and other statistics.
pop_sizes = {'sim1': 1402817, 'av2015': 1462158, 'sim2': 2371686, 'av2017': 2483089}


def compute_proportions(pair, comparison_table):
    r"""Compute category sizes as a proportion of the total population from value counts data for a pair of populations.
    
    Augments the input table of counts data with columns for proportion by category and returns the result.
    """
    table = comparison_table.copy()
    table['proportion_'+pair[0]] = table['counts_'+pair[0]]/pop_sizes[pair[0]]
    table['proportion_'+pair[1]] = table['counts_'+pair[1]]/pop_sizes[pair[1]]
    return table


def compute_z_test(pair, comparison_table):
    r"""Compute z-test statistics based on value counts data from two populations.
    
    Augments the input table of counts data with columns for z-test statistics, and intermediate calculated values 
    including proportion by category, and difference in observed category proportions between populations;
    and returns the resulting table.
    """
    table = compute_proportions(pair, comparison_table)
    table['p_diff'] = table['proportion_'+pair[0]] - table['proportion_'+pair[1]]
    table['p_ave'] = (table['counts_'+pair[0]] + table['counts_'+pair[1]]) / (pop_sizes[pair[0]] + pop_sizes[pair[1]])
    table['z_test'] = table['p_diff']/np.sqrt(table['p_ave'] * (1 - table['p_ave']) * ((1/pop_sizes[pair[0]]) + (1/pop_sizes[pair[1]])))
    return table


def compute_chi2_test(pair, comparison_table, grouping='univariate'):
    r"""Compute Pearson's chi-squared test statistics based on value counts data from two populations.
    
    Returns a pandas DataFrame indexed by category name, with columns for chi-squared test results, number of degrees of freedom, 
    category size, and a normalized score based on the number of degrees of freedom. 
    The normalized score is computed using a Wilson–Hilferty transformation, which approximately normalizes a chi-squared distribution.
    Under the null hypothesis, Pearson's chi-squared test statistic follows a chi-squared distribution with the given number of degrees of freedom, and so the Wilson–Hilferty transformation of the statistic should be distributed as Normal(0,1).
    """
    assert grouping in ['univariate', 'bivariate']
    # Augments the input table of counts data with columns for proportion by category
    table = compute_proportions(pair, comparison_table)
    # Calculate the expected frequency by category based on proportion in the reference population 
    table['expected_count_'+pair[0]] = pop_sizes[pair[0]] * table['proportion_'+pair[1]]
    # Calculate the squared error between the observed frequency (counts) and the expected frequency.
    table['squared_error'] = np.square(table['counts_'+pair[0]] - table['expected_count_'+pair[0]])
    # Calculate a summand of the Pearson's chi-squared test statistic for each category, avoiding divide-by-zero errors
    table['pearson_chi2_test'] = table['squared_error'] / table['expected_count_'+pair[0]].where(table['expected_count_'+pair[0]] != 0, other=0.5)
    # Group the summands by category ahead of aggregation (summation). Categories are field pairs in the bivariate case.
    if grouping == 'univariate':
        group_cols = ['column_name']
    elif grouping == 'bivariate':
        group_cols = ['column_name1', 'column_name2']
    grouped = table[group_cols + ['pearson_chi2_test']].groupby(by=group_cols)
    # Aggregate the data and rename the columns
    results = pd.concat([grouped.size(), grouped['pearson_chi2_test'].sum(), grouped.size() - 1], axis=1)
    results.columns = ['category_size', 'pearson_chi2_test', 'degrees_of_freedom']
    # Normalize the chi-squared test statistics by subtracting the mean and dividing by standard deviation
    results['normalized_score'] = (results['pearson_chi2_test'] - results['degrees_of_freedom']) / np.sqrt(2 * results['degrees_of_freedom'])
    # Apply the Wilson–Hilferty transformation to the chi-squared test statistics to approximately normalize them (under null hypothesis)
    results['Wilson–Hilferty_score'] = (np.cbrt(results['pearson_chi2_test']/results['degrees_of_freedom'])
                                   - (1 - 2/(9 * results['degrees_of_freedom']))) / (2/(9 * results['degrees_of_freedom']))
    return results


def compute_cdf(pair, comparison_table, grouping='univariate'):
    r"""Compute cumulative distribution functions (CDF) based on ordered value counts data for a pair of populations.
    
    Augments the input table of counts data with columns for proportion by data value, groups and sorts the data by field and date/age,
    then computes the cumulative distribution function for each field as the cumulative sum of proportion, adds these columns to the 
    augmented table and returns the result.
    """
    assert grouping in ['univariate', 'bivariate']
    table = compute_proportions(pair, comparison_table)
    # Group and sort the data by field ahead of cumulative summation.
    if grouping == 'univariate':
        group_cols = ['column_name']
        sort_order = ['column_name', 'val']
    elif grouping == 'bivariate':
        group_cols = ['column_name1', 'val1']
        sort_order = ['column_name1', 'val1', 'val2']
    table = table.sort_values(by=sort_order)
    # Compute the cumulative distribution function for each field as the cumulative sum of proportion values
    cumsum_results = table[group_cols+['proportion_'+key for key in pair]].groupby(by=group_cols).cumsum()
    cumsum_results.columns = ['cdf_'+key for key in pair]
    # Add the CDF values to our table
    results = pd.concat([table, cumsum_results], axis=1)
    return results


def compute_ks_test(pair, comparison_table, grouping='univariate'):
    r"""Compute Kolmogorov-Smirnov test statistics based on ordered value counts data by field from two populations.
    
    Returns a pandas DataFrame indexed by field name, with columns for KS-test results, critical values, and p-values
    The critical values are obtained by scaling the KS-test statistic by a factor relating to the sizes of the two populations.
    The p-values are obtained by a transformation of the critical values.
    """
    cdf_table = compute_cdf(pair, comparison_table, grouping)
    cdf_table['cdf_diff'] = (cdf_table['cdf_'+pair[0]] - cdf_table['cdf_'+pair[1]]).abs()
    # Group the data by field and find the maximum absolute vertical difference between CDFs for each field.
    if grouping == 'univariate':
        group_cols = ['column_name']
    elif grouping == 'bivariate':
        group_cols = ['column_name1', 'val1']
    results = cdf_table[group_cols + ['cdf_diff']].groupby(by=group_cols).max().dropna()
    results.columns = ['ks_test_statistic']
    results['ks_scaled'] = results['ks_test_statistic'] * np.sqrt((pop_sizes[pair[0]] * pop_sizes[pair[1]])/(pop_sizes[pair[0]] + pop_sizes[pair[1]]))
    results['p_value'] = np.exp(-2 * np.square(results['ks_scaled']))
    return results


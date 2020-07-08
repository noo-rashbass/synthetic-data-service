#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file can be imported as a module and contains the following functions:
    * plot_univariate_categorical_results - Plot grouped bar charts of z-test statistics by field values
    * plot_bivariate_categorical_results - Plot heatmaps of z-test statistics by field value pairs
    * make_hovertext - Create a Series of hovertext strings for a list of columns in a DataFrame for use in Plotly
    * plot_univariate_chi2_test_results - Plot grouped bar charts of chi-squared test statistics by field
This module also contains the parameter `marker_colour` which is a dictionary defining how bar charts are coloured based on the pair of source cohort tables being compared.
"""

# Third-party imports
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Local packages
from params import field_list_dict


__author__ = 'Edward Pearce'
__copyright__ = 'Copyright 2019, Simulacrum Test Suite'
__credits__ = ['Edward Pearce']
__license__ = 'MIT'
__version__ = '1.0.0'
__maintainer__ = 'Edward Pearce'
__email__ = 'edward.pearce@phe.gov.uk'
__status__ = 'Development'


# This variable defines how bar charts are coloured for each pair of source cohort tables being compared
marker_colour = {('sim1', 'av2015'): 'blue', ('sim2', 'av2017'): 'lightskyblue'}


def plot_univariate_categorical_results(results_dict, col_name):
    r"""Produces grouped bar charts of z-test statistics by field values.

    Produces interactive grouped bar charts of z-test statistics based on value counts in a given field.
    One group of bar charts is drawn for each table of z-test results (pairs of source tables being compared) passed in the input dictionary.

    Parameters
    ----------
    results_dict : dictionary of pandas DataFrame objects
        The dataframes of z-test results from which we draw the data to plot
    col_name : str
        The name of the column in our dataframe we use to obtain the values for the x-axis.

    Returns
    -------
    A plotly.graph_objects Figure
        The Figure object containing our plot

    """
    fig = go.Figure()
    # For each pair of source tables to compare, plot the z-test statistics by value in a given field, highlighting high absolute values
    for pair, comparison_table in results_dict.items():
        frame = comparison_table.query("column_name == '{}'".format(col_name))
        m = frame.z_test.abs() < 2
        fig.add_trace(go.Bar(name=pair[0]+' vs. '+pair[1], x=frame.val, y=frame.z_test,
                             marker_color=marker_colour[pair],
                             marker_line_width=frame.z_test.mask(m, 0).mask(~m, 1),
                             marker_line_color=frame.z_test.mask(m, marker_colour[pair]).mask(~m, 'red')))
    # Change the bar mode, set axis titles
    fig.update_layout(barmode='group', xaxis_title=col_name, yaxis_title='z-test statistic', title='Univariate z-test results')
    # Plot columns which are not 'AGE' as categories to avoid automatically being cast as numerical values
    if col_name != 'AGE':
        fig.update_layout(xaxis={'type': 'category', 'categoryorder':'category ascending'})
    return fig


def plot_bivariate_categorical_results(results_dict, col_name1, col_name2, display=True):
    r"""Produces heatmaps of z-test statistics by field value pairs.

    Produces a dictionary of interactive heatmaps of z-test statistics based on value counts in field value pairs for the given pair of data fields. 
    One heatmap is created for each table of z-test results passed in the input dictionary.

    Parameters
    ----------
    results_dict : dictionary of pandas DataFrame objects
        The dataframes of z-test results from which we draw the data to plot
    col_name1 : str
        The name of the column in our dataframe we use to obtain the values for the x-axis.
    col_name2 : str
        The name of the column in our dataframe we use to obtain the values for the y-axis.
    display : Boolean. Defaults to True
        Set display to False to return the dictionary of heatmaps directly, otherwise running the function immediately displays the plots upon completion.

    Returns
    -------
    A dictionary of plotly.graph_objects Figures
        The Figure objects containing our plots, one per table of z-test results

    """
    # Create a plot for each table of z-test statistics comparing two source tables
    fig = {pair: go.Figure() for pair in results_dict.keys()}
    for pair, comparison_table in results_dict.items():
        # Locate the rows of the comparison table where the column names match our inputs
        row_mask = comparison_table[['column_name1', 'column_name2']].isin([col_name1, col_name2]).all(axis=1)
        frame = comparison_table.loc[row_mask, ['val1', 'val2', 'z_test']]
        # These conditional statements ensure that `col_name1` is plotted on the x-axis, and `col_name2` on the y-axis
        if (col_name1, col_name2) in field_list_dict['bivariate_categorical']:
            x_val_col, y_val_col = 'val1', 'val2'
        elif (col_name2, col_name1) in field_list_dict['bivariate_categorical']:
            x_val_col, y_val_col = 'val2', 'val1'
        else:
            print('This column name pair cannot be found')    
            return
        # Obtain the lists of unique x-values and y-values
        frame = frame.sort_values(by=[x_val_col, y_val_col])
        x_vals = frame[x_val_col].drop_duplicates()
        y_vals = frame[y_val_col].drop_duplicates()
        # The z-values are used to colour the heatmap and are stored in a 2D-array. Rows for y-values, columns for x-values.
        z_vals = list()        
        for y_val in y_vals:
            z_row = list()
            y_subframe = frame.loc[frame[y_val_col] == y_val, [x_val_col, 'z_test']]
            for x_val in x_vals:
                x_y_subframe = y_subframe.loc[y_subframe[x_val_col] == x_val, 'z_test']
                # Check that there is any data (a z-test statistic) for the given x,y-pair
                if x_y_subframe.shape[0] == 1:
                    # If there is, then add it to our array
                    z_row.append(x_y_subframe.iloc[0])
                else:
                    # Otherwise add a null value which represents a structural zero present in both source datasets
                    z_row.append(np.nan)
            z_vals.append(z_row) 
        # Plot the heatmap based on the x,y,z-values we extracted from the results table
        fig[pair].add_trace(go.Heatmap(name=pair[0]+' vs. '+pair[1], z=z_vals, x=x_vals, y=y_vals, 
                                       colorscale='Picnic', zmin=-7, zmax=7, colorbar={"title": "z-test statistic"}))
        # Default behaviour to set axes and layout
        fig[pair].update_layout(title=pair[0]+' vs. '+pair[1],
                                xaxis={'title': col_name1, 'type': 'category', 'categoryorder':'category ascending'}, 
                                yaxis={'title': col_name2, 'type': 'category', 'categoryorder':'category descending'})
        # Special axis settings to order ages numerically rather than alphanumerically as strings
        if col_name1 == 'AGE':
            fig[pair].update_xaxes({'categoryorder': 'array', 
                                    'categoryarray': [str(number) for number in sorted(x_vals.astype('uint8'))]})
        elif col_name2 == 'AGE':
            fig[pair].update_yaxes({'categoryorder': 'array', 
                                    'categoryarray': [str(number) for number in sorted(y_vals.astype('uint8'))]})
    if display == True:
    # Display the plots!
        for key, plot in fig.items():
            plot.show()
        return
    else:
        return fig


def make_hovertext(table, column_labels):
    r"""Create a Series of hovertext strings for a list of columns in a DataFrame for use in Plotly.
    
    Returns a Series of the same length as the input DataFrame containing hovertext strings detailing the values in the passed list of columns
    """
    string_series = pd.concat([table[label].apply(lambda val: '{}={}<br>'.format(label, val)) for label in column_labels], axis=1).sum(axis=1)
    return string_series


def plot_univariate_chi2_test_results(results_dict, by='Wilson–Hilferty_score'):
    r"""Produces grouped bar charts of chi-squared test statistics by field.

    Produces interactive grouped bar charts of chi-squared test statistics based on value counts in each field.
    One group of bar charts is drawn for each table of chi-squared test results (pairs of source tables being compared) passed in the input dictionary.

    Parameters
    ----------
    results_dict : dictionary of pandas DataFrame objects
        The dataframes of chi-squared test results from which we draw the data to plot

    Returns
    -------
    A plotly.graph_objects Figure
        The Figure object containing our plot

    """
    fig = go.Figure()
    # For each pair of source tables to compare, plot a horizontal bar chart of the chi-squared test statistics
    # 
    for pair, results_table in results_dict.items():
        fig.add_trace(go.Bar(name=pair[0]+' vs. '+pair[1],
                             orientation='h',
                             x=results_table[by],
                             y=results_table.index, 
                             marker_color=marker_colour[pair],
                             hoverinfo="name+y+text",
                             hovertext=make_hovertext(results_table, ['pearson_chi2_test', 'degrees_of_freedom', 'normalized_score', 'Wilson–Hilferty_score'])))
    # Change the bar mode, set axis titles, and sort the bar lengths
    fig.update_layout(barmode='group', 
                      xaxis_title=by, 
                      yaxis_title='Field name', 
                      title='Univariate chi-squared test results',
                      xaxis_type='log',
                      yaxis_categoryorder='total ascending')
    return fig


def plot_bivariate_chi2_results(results_dict, by='Wilson–Hilferty_score', display=True):
    r"""Produces heatmaps of chi-squared test statistics by field value pairs.

    Produces a dictionary of interactive heatmaps of chi-squared test statistics based on category sizes for categories defined by values in a pair of data fields. 
    One heatmap is created for each table of chi-squared test results passed in the input dictionary.

    Parameters
    ----------
    results_dict : dictionary of pandas DataFrame objects
        The dataframes of chi-squared test results from which we draw the data to plot
    display : Boolean. Defaults to True
        Set display to False to return the dictionary of heatmaps directly, otherwise running the function immediately displays the plots upon completion.
        
    Returns
    -------
    A dictionary of plotly.graph_objects Figures
        The Figure objects containing our plots, one per table of chi-squared test results

    """
    # Create a plot for each table of chi-squared test statistics comparing two source tables
    fig = {pair: go.Figure() for pair in results_dict.keys()}
    for pair, results_table in results_dict.items():
        # Create the hovertext strings for each pair of fields
        hovertext_table = make_hovertext(results_table, ['pearson_chi2_test', 'degrees_of_freedom', 'normalized_score', 'Wilson–Hilferty_score'])
        # Reset indices to make it easier to select the data
        results_table = results_table.reset_index()
        hovertext_table = hovertext_table.reset_index()
        # Get the list of values for the x,y-axes
        x_vals = y_vals = field_list_dict['univariate_categorical']
        # The z-values are used to colour the heatmap and are stored in a 2D-array. Rows for y-values, columns for x-values.
        z_vals = list()
        hovertext_array = list()
        for y_val in y_vals:
            z_row = list()
            hovertext_row = list()
            for x_val in x_vals:
                string_query = "column_name1 in ['{0}', '{1}'] and column_name2 in ['{0}', '{1}']".format(x_val, y_val)
                x_y_subframe = results_table.query(string_query)                
                # Equivalently, ...                
#                 row_mask = results_table[['column_name1', 'column_name2']].isin([x_val, y_val]).all(axis=1)
#                 x_y_subframe = results_table.loc[row_mask]
                # Check that there is any data for the given x,y-pair
                if x_y_subframe.shape[0] == 1:
                    # If there is, then add it to our array
                    z_row.append(x_y_subframe[by].iloc[0])
                    hovertext_row.append(hovertext_table.query(string_query)[0].iloc[0])
                else:
                    # Otherwise add a null value
                    z_row.append(np.nan)
                    hovertext_row.append("")
            z_vals.append(z_row)
            hovertext_array.append(hovertext_row)
        # Plot the heatmap based on the x,y,z-values we extracted from the results table
        fig[pair].add_trace(go.Heatmap(name=pair[0]+' vs. '+pair[1], z=z_vals, x=x_vals, y=y_vals, 
                                       hovertext=hovertext_array,                                       
                                       colorscale=[
                                           [0, 'rgb(247, 251, 255)'],
                                           [1./100000, 'rgb(222, 235, 247)'],
                                           [1./20000, 'rgb(198, 219, 239)'],
                                           [1./10000, 'rgb(158, 202, 225)'],
                                           [1./5000, 'rgb(107, 174, 214)'],
                                           [1./1000, 'rgb(66, 146, 198)'],
                                           [1./100, 'rgb(33, 113, 181)'],
                                           [1./10, 'rgb(8, 81, 156)'],
                                           [1., 'rgb(8, 48, 107)']], 
                                       zmin=0, zmax=100000, 
                                       colorbar={'title': by, 
                                                 'tick0': 0,
                                                 'tickmode': 'array',
                                                 'tickvals': [0, 10, 100, 1000, 5000, 10000, 20000, 100000]
                                                }))
        # Set graph and axis titles
        fig[pair].update_layout(title=pair[0]+' vs. '+pair[1],
                                xaxis_title='Field 1', 
                                yaxis_title='Field 2')
    if display == True:
    # Display the plots!
        for key, plot in fig.items():
            plot.show()
        return
    else:
        return fig

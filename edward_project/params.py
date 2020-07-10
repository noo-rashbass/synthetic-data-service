#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file contains the following module parameters:
    * params - Connection parameters for the database.connect function
    * filepath_dictionary - Contains names of filepaths where local copies of group counts data can be stored
    
Column name related parameters, mostly encapsulated in the variable `field_list_dict` which stores various lists of column names and pairs of column names:
    * categorical_cols - A list of non-index column names for categorical/discrete value fields in SIM_AV_TUMOUR, plus two derived categorical fields.
    * date_cols - A list of non-index column names for date value fields in SIM_AV_TUMOUR.
    * col_names - The concatenation of the two lists `categorical_cols` and `date_cols`.
    * col_name_pairs - List of pairs of non-index columns present in SIM_AV_TUMOUR in the Simulacrum.
    * categorical_col_pairs - 
    * category_cross_date_pairs - 
    
    * plot_params_dict - A dictionary of plotting parameters used by the plots.plot_by_category function. The keys are plot_type options and the values are dictionaries of plotting parameters.
"""

# Parameters for the database.connect function
params = dict(dialect='oracle',
              driver='cx_oracle',
              server='localhost',
              port=1523,
              database='casref01')


# Aliases for tables of CAS/Simulacrum data
key_list = ['sim2', 'av2017'] # Same values as pop_queries.keys()
# Pairs of tables which will be compared
comparison_pairs = [('sim2', 'av2017')]

# Hard-coded list of non-index columns present in SIM_AV_TUMOUR in the Simulacrum. These are the options for the col_name function parameter.
categorical_cols = ['QUINTILE_2015', 'CREG_CODE', 'GRADE', 'SEX', 'SITE_ICD10_O2', 'SITE_ICD10_O2_3CHAR', 'MORPH_ICD10_O2', 'BEHAVIOUR_ICD10_O2', 'T_BEST', 'N_BEST', 'M_BEST', 'STAGE_BEST', 'STAGE_BEST_SYSTEM', 'SCREENINGSTATUSFULL_CODE', 'ER_STATUS', 'ER_SCORE', 'PR_STATUS', 'PR_SCORE', 'HER2_STATUS', 'LATERALITY', 'GLEASON_PRIMARY', 'GLEASON_SECONDARY', 'GLEASON_TERTIARY', 'GLEASON_COMBINED', 'CANCERCAREPLANINTENT', 'PERFORMANCESTATUS', 'CNS', 'ACE27', 'DIAGNOSISMONTHBEST', 'MONTH_FIRST_SURGERY', 'AGE',
'RADIOTHERAPYPRIORITY', 'RADIOTHERAPYINTENT', 'PRESCRIBEDDOSE', 'PRESCRIBEDFRACTIONS', 'ACTUALDOSE', 'TREATMENTREGION', 'TREATMENTANATOMICALSITE']

date_cols = ['DIAGNOSISDATEBEST', 'DATE_FIRST_SURGERY', 'EARLIESTCLINAPPROPDATE', 'DECISIONTOTREATDATE']

col_names = categorical_cols + date_cols

# List of pairs of non-index columns present in SIM_AV_TUMOUR in the Simulacrum.
col_name_pairs = [(col_names[i], col_names[j]) for i in range(len(col_names)) for j in range(i+1, len(col_names))]

categorical_col_pairs = [(categorical_cols[i], categorical_cols[j]) 
                         for i in range(len(categorical_cols)) 
                         for j in range(i+1, len(categorical_cols))]

category_cross_date_pairs = [(categorical_col, date_col)
                            for date_col in date_cols
                            for categorical_col in categorical_cols]

field_list_dict = {'univariate_categorical': categorical_cols,
                   'univariate_dates': date_cols,
                   'bivariate_categorical': categorical_col_pairs,
                   'categorical_cross_diagnosis_date': [(categorical_col, 'DIAGNOSISDATEBEST') for categorical_col in categorical_cols],
                   'categorical_cross_surgery_date': [(categorical_col, 'DATE_FIRST_SURGERY') for categorical_col in categorical_cols],
                   'surgery_date_cross_diagnosis_date': [('DATE_FIRST_SURGERY', 'DIAGNOSISDATEBEST')]
                   }


# File names used for storing, retrieving local copies of grouped counts data extracted from the SQL database
filepath_templates = {'univariate_categorical': r"E:\{}_univariate_categorical.csv",
                      'univariate_dates': r"E:\{}_univariate_dates.csv",
                      'bivariate_categorical': r"E:\{}_bivariate_categorical.csv",
                      'categorical_cross_diagnosis_date': r"E:\{}_categorical_cross_diagnosis_dates.csv",
                      'categorical_cross_surgery_date': r"E:\{}_categorical_cross_surgery_dates.csv",
                      'surgery_date_cross_diagnosis_date': r"E:\{}_bivariate_counts_double_dates.csv"
                      }

filepath_dictionary = {count_type: {key: template.format(key.upper()) for key in ['sim2', 'av2017']} for count_type, template in filepath_templates.items()}


# Hard-coded dictionary of plotting parameters used by the plot_by_category function. 
# The keys are plot_type options and the values are dictionaries of plotting parameters.
plot_params_dict = {'Counts': 
                    {'y_values': ['counts_r', 'counts_s'],
                    'labels': ['Real', 'Simulated'],
                    'x_label': 'Category',
                    'y_label': 'Counts',
                    'title': 'Counts by Category in {}'},
                'Proportions': 
                    {'y_values': ['proportion_r', 'proportion_s'],
                    'labels': ['Real', 'Simulated'],
                    'x_label': 'Category',
                    'y_label': 'Proportion',
                    'title': 'Proportion by Category in {}'},
                'Absolute Difference': 
                    {'y_values': ['abs_diff'],
                    'labels': ['$p_{sim} - p_{real}$'],
                    'x_label': 'Category',
                    'y_label': 'Difference in Proportions',
                    'title': 'Difference in Proportions by Category in {}'},
                'Relative Difference': 
                    {'y_values': ['rel_diff'],
                    'labels': ['$(p_{sim} - p_{real})/p_{real}$'],
                    'x_label': 'Category',
                    'y_label': 'Relative Difference in Proportions',
                    'title': 'Relative Difference in Proportions by Category in {}'},
                'One-sample Binomial z-test': 
                    {'y_values': ['binom_z_test_one_sample'],
                    'labels': ['$z = (X - np)/\sqrt{np(1-p)}$'],
                    'x_label': 'Category',
                    'y_label': 'z-test statistic',
                    'title': 'Binomial One-sample z-test statistic by Category in {}'},
                'Pooled Two-sample Binomial z-test': 
                    {'y_values': ['z_test_two_sample_pooled'],
                    'labels': ['$z = (p_{1}-p_{2})/\sqrt{\hat{p}(1-\hat{p})(1/n_{1}+1/n_{2})}$'],
                    'x_label': 'Category',
                    'y_label': 'z-test statistic',
                    'title': 'Pooled Two-sample Binomial z-test statistic by Category in {}'},
                'Pearson Chi-squared test summand': 
                    {'y_values': ['pearson_summand'],
                    'labels': ['$(O - E)^{2}/E$'],
                    'x_label': 'Category',
                    'y_label': 'Pearson test summand',
                    'title': 'Pearson Chi-squared test summand by Category in {}'},
                'Likelihood-ratio test summand': 
                     {'y_values': ['lr_summand'],
                    'labels': ['$O * \ln(O/E)$'],
                    'x_label': 'Category',
                    'y_label': 'Likelihood-ratio test summand',
                    'title': 'Likelihood-ratio test summand by Category in {}'}
                    }

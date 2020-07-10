#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file can be imported as a module and contains module parameters.
In particular it contains a Python dictionary called `pop_queries` which contains SQL queries which can be run on the CAS
to generate extracts of real and simulated tumour data prepped for comparison.

The tumour data queries contained in the Python dictionary `pop_queries` are as follows:
    * SIM1_pop_query - SQL query to construct a table of tumour data from Simulacrum version 1 (SIM_AV_TUMOUR_FINAL)
    * AV2015_pop_query - A string containing an SQL query to construct a table of tumour data from the AV2015 snapshot
    * SIM2_pop_query - SQL query to construct a table of tumour data from Simulacrum version 2 (SIM_AV_TUMOUR_simII)
    * AV2017_pop_query - A string containing an SQL query to construct a table of tumour data from the AV2017 snapshot
    
It should be noted that the following preprocessing steps are taken when extracting the data:
  - Strip extra text from `QUINTILE_2015`, retaining only Deprivation Index score (integer between 1 and 5 inclusive)
  - Strip the first character (alphabetical) from `CREG_CODE` (Cancer Registry Code), retaining only a 4 digit numerical code string
  - Put `DIAGNOSISDATEBEST`, `DATE_FIRST_SURGERY` into YYYY-MM-DD format for easier sorting
  - Introducing derived fields `DIAGNOSISMONTHBEST`, `MONTH_FIRST_SURGERY` in YYYY-MM format for analysis as discrete variables
"""


__author__ = 'Edward Pearce'
__copyright__ = 'Copyright 2019, Simulacrum Test Suite'
__credits__ = ['Edward Pearce']
__license__ = 'MIT'
__version__ = '1.0.0'
__maintainer__ = 'Edward Pearce'
__email__ = 'edward.pearce@phe.gov.uk'
__status__ = 'Development'


# A string containing an SQL query to construct a table of tumour data from Simulacrum version 1 (SIM_AV_TUMOUR_FINAL)
# The corresponding table of real tumour data is drawn from the AV2015 snapshot, with diagnosis dates between 2013-01-01 and 2015-12-31
SIM1_pop_query = '''SELECT
SUBSTR(QUINTILE_2015, 1, 1) AS QUINTILE_2015,
SUBSTR(CREG_CODE, 2) AS CREG_CODE,
TO_CHAR(DIAGNOSISDATEBEST, 'YYYY-MM-DD') AS DIAGNOSISDATEBEST,
TO_CHAR(DIAGNOSISDATEBEST, 'YYYY-MM') AS DIAGNOSISMONTHBEST,
TO_CHAR(DATE_FIRST_SURGERY, 'YYYY-MM-DD') AS DATE_FIRST_SURGERY,
TO_CHAR(DATE_FIRST_SURGERY, 'YYYY-MM') AS MONTH_FIRST_SURGERY,
AGE, GRADE, SEX,
SITE_ICD10_O2, SITE_ICD10_O2_3CHAR, MORPH_ICD10_O2, BEHAVIOUR_ICD10_O2,
T_BEST, N_BEST, M_BEST, STAGE_BEST, STAGE_BEST_SYSTEM,
SCREENINGSTATUSFULL_CODE, ER_STATUS, ER_SCORE, PR_STATUS, PR_SCORE, HER2_STATUS, LATERALITY,
GLEASON_PRIMARY, GLEASON_SECONDARY, GLEASON_TERTIARY, GLEASON_COMBINED,
CANCERCAREPLANINTENT, PERFORMANCESTATUS, CNS, ACE27
FROM analysispaulclarke.sim_av_tumour_final
'''


# A string containing an SQL query to construct a table of tumour data from the AV2015 snapshot
# ready for comparison with its Simulacrum counterpart (Version 1)
AV2015_pop_query = '''SELECT
SUBSTR(multi_depr_index.QUINTILE_2015, 1, 1) AS QUINTILE_2015,
SUBSTR(av_tumour.CREG_CODE, 2) AS CREG_CODE,
TO_CHAR(av_tumour.DIAGNOSISDATEBEST, 'YYYY-MM-DD') AS DIAGNOSISDATEBEST,
TO_CHAR(av_tumour.DIAGNOSISDATEBEST, 'YYYY-MM') AS DIAGNOSISMONTHBEST,
TO_CHAR(av_tumour_exp.DATE_FIRST_SURGERY, 'YYYY-MM-DD') AS DATE_FIRST_SURGERY,
TO_CHAR(av_tumour_exp.DATE_FIRST_SURGERY, 'YYYY-MM') AS MONTH_FIRST_SURGERY,
av_tumour.AGE, av_tumour.GRADE, av_tumour.SEX,
av_tumour.SITE_ICD10_O2, av_tumour.SITE_ICD10_O2_3CHAR, av_tumour.MORPH_ICD10_O2, av_tumour.BEHAVIOUR_ICD10_O2,
av_tumour.T_BEST, av_tumour.N_BEST, av_tumour.M_BEST, av_tumour.STAGE_BEST, av_tumour.STAGE_BEST_SYSTEM,
av_tumour.SCREENINGSTATUSFULL_CODE, av_tumour.ER_STATUS, av_tumour.ER_SCORE, av_tumour.PR_STATUS, av_tumour.PR_SCORE, av_tumour.HER2_STATUS, av_tumour.LATERALITY,
av_tumour_exp.GLEASON_PRIMARY, av_tumour_exp.GLEASON_SECONDARY, av_tumour_exp.GLEASON_TERTIARY, av_tumour_exp.GLEASON_COMBINED,
av_tumour_exp.CANCERCAREPLANINTENT, av_tumour_exp.PERFORMANCESTATUS, av_tumour_exp.CNS, av_tumour_exp.ACE27
FROM
(SELECT TUMOURID, LSOA11_CODE, GRADE, AGE, SEX, CREG_CODE, SCREENINGSTATUSFULL_CODE, ER_STATUS, ER_SCORE, PR_STATUS, PR_SCORE, HER2_STATUS, LATERALITY, DIAGNOSISDATEBEST, SITE_ICD10_O2, SITE_ICD10_O2_3CHAR, MORPH_ICD10_O2, BEHAVIOUR_ICD10_O2, T_BEST, N_BEST, M_BEST, STAGE_BEST, STAGE_BEST_SYSTEM
FROM AV2015.AV_TUMOUR
WHERE (DIAGNOSISDATEBEST BETWEEN '01-JAN-2013' AND '31-DEC-2015') AND STATUSOFREGISTRATION = 'F' AND CTRY_CODE = 'E' AND DEDUP_FLAG = 1) av_tumour
LEFT JOIN
(SELECT TUMOURID, CANCERCAREPLANINTENT, PERFORMANCESTATUS, CNS, ACE27, DATE_FIRST_SURGERY, GLEASON_PRIMARY, GLEASON_SECONDARY, GLEASON_TERTIARY, GLEASON_COMBINED 
FROM AV2015.AV_TUMOUR_EXPERIMENTAL_1612) av_tumour_exp
ON av_tumour.tumourid = av_tumour_exp.tumourid
LEFT JOIN IMD.ID2015 multi_depr_index
ON av_tumour.LSOA11_CODE = multi_depr_index.LSOA11_CODE
'''


# A string containing an SQL query to construct a table of tumour data from Simulacrum version 2 (SIM_AV_TUMOUR_simII)
# The corresponding table of real tumour data is drawn from the AV2017 snapshot, with diagnosis dates between 2013-01-01 and 2017-12-31
SIM2_pop_query = '''SELECT
SUBSTR(QUINTILE_2015, 1, 1) AS QUINTILE_2015,
SUBSTR(CREG_CODE, 2) AS CREG_CODE,
TO_CHAR(DIAGNOSISDATEBEST, 'YYYY-MM-DD') AS DIAGNOSISDATEBEST,
TO_CHAR(DIAGNOSISDATEBEST, 'YYYY-MM') AS DIAGNOSISMONTHBEST,
TO_CHAR(DATE_FIRST_SURGERY, 'YYYY-MM-DD') AS DATE_FIRST_SURGERY,
TO_CHAR(DATE_FIRST_SURGERY, 'YYYY-MM') AS MONTH_FIRST_SURGERY,
AGE, GRADE, SEX,
SITE_ICD10_O2, SITE_ICD10_O2_3CHAR, MORPH_ICD10_O2, BEHAVIOUR_ICD10_O2,
T_BEST, N_BEST, M_BEST, STAGE_BEST, STAGE_BEST_SYSTEM,
SCREENINGSTATUSFULL_CODE, ER_STATUS, ER_SCORE, PR_STATUS, PR_SCORE, HER2_STATUS, LATERALITY,
GLEASON_PRIMARY, GLEASON_SECONDARY, GLEASON_TERTIARY, GLEASON_COMBINED,
CANCERCAREPLANINTENT, PERFORMANCESTATUS, CNS, ACE27,

RADIOTHERAPYPRIORITY, RADIOTHERAPYINTENT, PRESCRIBEDDOSE, PRESCRIBEDFRACTIONS, ACTUALDOSE, TREATMENTREGION, TREATMENTANATOMICALSITE,
TO_CHAR(DECISIONTOTREATDATE, 'YYYY-MM-DD') AS DECISIONTOTREATDATE,
TO_CHAR(DECISIONTOTREATDATE, 'YYYY-MM') AS DECISIONTOTREATMONTH,
TO_CHAR(EARLIESTCLINAPPROPDATE, 'YYYY-MM-DD') AS EARLIESTCLINAPPROPDATE,
TO_CHAR(EARLIESTCLINAPPROPDATE, 'YYYY-MM') AS EARLIESTCLINAPPROPMONTH
FROM analysispaulclarke.sim_av_tumour_simII
INNER JOIN analysispaulclarke.sim_rtdsprescription_simII
on sim_rtdsprescription_simII.tumourid = sim_av_tumour_simII.tumourid
'''


# A string containing an SQL query to construct a table of tumour data from the AV2017 snapshot 
# ready for comparison with its Simulacrum counterpart (Version 2)
AV2017_pop_query = '''
select * from testing
'''


# We store all of our population queries into a Python dictionary data structure for convenient access
pop_queries = {'sim2': SIM2_pop_query.replace('\n', ' '),
               'av2017': AV2017_pop_query.replace('\n', ' ')}

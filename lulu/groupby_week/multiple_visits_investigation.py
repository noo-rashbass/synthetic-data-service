import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

## Import observations here: --------------------------------------------------------------------------
df = pd.read_csv('ISASimple_ICEMR_PRISM_cohort_RSRC_observations.txt', delimiter = '\t')
# df = df[['Participant_Id', 'Visit date [EUPATH_0000091]']] # pick features
df = df.dropna(subset=['Visit date [EUPATH_0000091]'])
df['Visit date [EUPATH_0000091]'] = pd.to_datetime(df['Visit date [EUPATH_0000091]'])

## Create visit week column containing date on MONDAY of the week the visit took place in: -------------
df['Visit week'] = df['Visit date [EUPATH_0000091]'].apply(lambda d: d-dt.timedelta(days=d.weekday()))
#print(df)

## Group by participant then by visit week: -----------------------------------------------------------
index = pd.MultiIndex.from_frame(df[['Participant_Id', 'Visit week']])
df = df.set_index(index)
# df = df.drop(columns=['Participant_Id', 'Visit week'])
df['counts'] = df.groupby(level=['Participant_Id', 'Visit week']).size()
# print(df)

########################################################################################################

## Get df for instances of 2 per week:
df_2 = df[df['counts']==2].drop(columns=['counts'])
# print(df_2.head(20))

## Get df for instances of 3 per week
df_3 = df[df['counts']==3].drop(columns=['counts'])
# print(df_3)

########################################################################################################

## What is the spread of multiple instances throughout the trail? --------------------------------------
## For the three category, instances are very isolated. It doesn't make for a very interesting histogram on its own. 
## But they might coincide with peaks in 2 visit instances
# print(df_3['Visit week'].unique())
# plt.hist(df_3['Visit week'].unique())
# plt.show()

# df_2_week_series = df_2['Visit week'].value_counts().sort_index() # beware over-counting, divide counts by two before viewing!
# df_2_weeks = df_2_week_series.index.tolist()
# df_2_weeks_freq = [x/2 for x in df_2_week_series.values.tolist()] # corrects over-counting :)
# plt.plot(df_2_weeks, df_2_weeks_freq)
# plt.show()

# df_2_week_list = df_2['Visit week'] # just the non-unique monday dates of all instances of 2 visits in that week (over-counted)
# # sns.kdeplot(df_2_week_list) ## DOES NOT WORK. CONVERT VISIT WEEK TO NUMBER OF WEEKS INTO THE STUDY
# plt.show()


## For individuals, do instances of 2 and 3 visits happen consecutively? (e.g. 5 visits in 2 weeks due to complications)


## Repeat above plots but split by scheduled and unscheduled -------------------------------------------
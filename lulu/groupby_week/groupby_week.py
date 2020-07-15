import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

## Import observations here: --------------------------------------------------------------------------
df = pd.read_csv('ISASimple_ICEMR_PRISM_cohort_RSRC_observations.txt', delimiter = '\t')
df = df[['Participant_Id', 'Visit date [EUPATH_0000091]']]
df = df.dropna()
df['Visit date [EUPATH_0000091]'] = pd.to_datetime(df['Visit date [EUPATH_0000091]'])

## Create visit week column containing date on MONDAY of the week the visit took place in: -------------
df['Visit week'] = df['Visit date [EUPATH_0000091]'].apply(lambda d: d-dt.timedelta(days=d.weekday()))
#print(df)

## Quick look at how this might smooth the data : ------------------------------------------------------
# (regular appointment intervals are still clearly visible, wich is good)
# totalperweek = df['Visit date [EUPATH_0000091]'].groupby(df['Visit week']).count().tolist()
# print(totalperweek)
# plt.plot(range(len(perweek)), perweek)
# plt.title('Total number of visits made by week')
# plt.show()

## Group by participant then by visit week: -----------------------------------------------------------
index = pd.MultiIndex.from_frame(df[['Participant_Id', 'Visit week']])
df = df.set_index(index)
df = df.drop(columns=['Participant_Id', 'Visit week'])
df['counts'] = df.groupby(level=['Participant_Id', 'Visit week']).size()
# print(df) # just to check
print('Most number of visits per week by a single participant: ', str(max(df['counts'])))
# print(df.apply(pd.value_counts))
# print(df['counts'].tolist()) # just to check the groupby is sensible

## Visualise dist of number of visits per week as table and as bar chart: -------------------------------------------------------
df['counts'] = pd.Categorical(df.counts) # make counts categorical
count_freqs = df['counts'].value_counts()
print('Instances of each number of visits per week (e.g. an individual who has 2 weeks with 3 visits adds 2 counts to the 3rd row of table: ')
print(count_freqs)

ax = plt.subplot()
plt.bar([1,2,3], count_freqs)
ax.set_xticks([1,2,3])
ax.set_xticklabels([1,2,3])
plt.xlabel('Number of Visits in a Week (Monday start)')
plt.ylabel('Instances')
plt.title('Instances of Each Number of Visits per Week')
plt.show()

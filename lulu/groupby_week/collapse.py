### UNDER CONTRUCTION ###

import pandas as pd
from column_group_maker import groups_dict, column_names # contains groupings by method of collapse
from multiple_visits_investigation import df, df_2, df_3 # for testing
from itertools import chain

df=df_2

df = df[df['Participant_Id']==1003]
df.reset_index(drop=True, inplace=True)
df = df.sort_values('Visit date [EUPATH_0000091]')
# index = pd.MultiIndex.from_frame(df[['Visit week', 'Visit date [EUPATH_0000091]']])
# df = df.set_index(index)
# df_2_1003 = df_2_1003.groupby(level=['Visit week', 'Visit date [EUPATH_0000091]'])
print(df)

columns = df.columns.tolist()

# ## CONVERT LISTLIKES TO SETS
# for col in groups_dict['ListLike']:
	# df[col] = df[col].apply(lambda x: x.split(' | ') if isinstance(x,str) else {})
	# print(df[col])

visit_weeks = list(df['Visit week'].unique())
df_week_collapsed = pd.DataFrame(columns=columns.remove('Observation_Id'), index=visit_weeks)
df_week_collapsed['Participant_Id']=1003
for week in visit_weeks:
	df_week = df[df['Visit week']==week]
	# print(df_week)
	for col in groups_dict['YesNo']:
		df_week_collapsed.loc[week, col] = ['Yes' if 'Yes' in df_week[col].values else ('No' if 'No' in df_week[col].values else 'Unable to assess')]
	for col in groups_dict['Continuous']:
		df_week_collapsed.loc[week, col] = df_week[col].mean()
	for col in groups_dict['Days']:
		df_week_collapsed.loc[week, col] = round(df_week[col].mean())
	for col in groups_dict['Text']:
		df_week_collapsed.loc[week, col] = ' | '.join(df_week[col].dropna().values)
	# for col in groups_dict['ListLike']:
	# 	l = [[value for value in df_week[col]]]
	# 	df_week_collapsed.loc[week, col] = l



# print(df_week_collapsed['Non-malaria medication [EUPATH_0000059]'])
print(len(df_week_collapsed.columns))
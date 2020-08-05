import pandas as pd
import numpy as np
from column_group_maker import groups_dict, column_names # contains groupings by method of collapse
from multiple_visits_investigation import df, df_2, df_3 # Already have visit week column containing the date on the Monday

def collapse_by_week(dataframe, filename):

	dataframe.reset_index(drop=True, inplace=True)

	COLLAPSED_DF = pd.DataFrame(columns=dataframe.columns)
	for i in dataframe['Participant_Id'].unique():

		df = dataframe.loc[dataframe['Participant_Id']==i]
		df = df.sort_values('Visit date [EUPATH_0000091]')

		columns = df.columns.tolist()

		visit_weeks = list(df['Visit week'].unique())

		df_week_collapsed = pd.DataFrame(columns=columns, index=visit_weeks)
		df_week_collapsed['Participant_Id']=i
		df_week_collapsed['Household_Id']=df['Household_Id'].iloc[0]
		for week in visit_weeks:
			df_week = df[df['Visit week']==week]
			# df_week_collapsed['visit week'] = week ############# THIS PREVIOUS VERSION GAVE THE REALLY STARANGE VISIT WEEK COL VALUES
			for col in groups_dict['YesNo']:
				# Alternative: encode numerically then take max
				df_week_collapsed.loc[week, col] = ['Yes' if 'Yes' in df_week[col].values else ('No' if 'No' in df_week[col].values else ('Unable to assess' if 'Unable to assess' in df_week[col].values else float('nan')))]
			for col in groups_dict['ContinuousAverage']:
				df_week_collapsed.loc[week, col] = df_week[col].mean()
			for col in groups_dict['RoundAverage']:
				df_week_collapsed.loc[week, col] = df_week[col].mean()
				if df_week_collapsed.loc[week, col] is not np.nan:
					df_week_collapsed.loc[week, col] = round(df_week_collapsed.loc[week, col])
			for col in groups_dict['JoinText']:
				df_week_collapsed.loc[week, col] = ' | '.join(df_week[col].dropna().values)
			for col in groups_dict['MakeList']:
				df_week[col] = df_week[col].dropna().apply(lambda x: str(x))
				df_week_collapsed.loc[week, col] = ' | '.join(df_week[col].dropna().values)

		df_week_collapsed.reset_index()
		COLLAPSED_DF = COLLAPSED_DF.append(df_week_collapsed, ignore_index=False) ############### PREVIOUSLY DROPPED INDEX AND BUILT VISIT WEEK ON LINE 25

	COLLAPSED_DF = COLLAPSED_DF[column_names]
	COLLAPSED_DF['visit week'] = COLLAPSED_DF.index ############## NEW: CONVERT CORRECT INDEX INTO VISIT WEEK COLUMN INSTEAD
	COLLAPSED_DF.to_csv(filename, index=False) 
	print(COLLAPSED_DF.head(10))
	return COLLAPSED_DF


## OUTPUTS
# df_1 = df[df['counts']==1].drop(columns=['counts'])
# df_1 = df_1[column_names]
# # df_1.to_csv('collapse_by_week_df_1.csv', index=False)

# collapse_by_week(df_2, 'collapse_by_week_df_2_with_week.csv')
# collapse_by_week(df_3, 'collapse_by_week_df_3_with_week2.csv')
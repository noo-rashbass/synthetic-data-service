import numpy as np
import pandas as pd


def pad(df, id, cols, start_date , end_date):
	duration_days = (end_date-start_date).days

	df = df[df['Participant_Id']==id]
	df = df[cols+['Visit date [EUPATH_0000091]']]

	df['day'] = df['Visit date [EUPATH_0000091]'].apply(lambda date: (date-start_date).days)
	df = df.sort_values(by='day')
	day_list = df['day'].tolist()
	df = df.drop(['Visit date [EUPATH_0000091]'], axis=1)

	df['day_next'] = df['day'].shift(-1)

	df = df.set_index('day')
	df.loc[max(day_list),'day_next'] = duration_days


	df_top = pd.DataFrame({'Participant_Id'})
	pdf = pd.DataFrame({'day': range(duration_days)})
	pdf = pdf.set_index('day')
	pdf = pdf.reindex(columns=cols)
	for d in day_list:
		for i in range(d, int(df.loc[d,'day_next'])):
			pdf.loc[i, cols]=df.loc[d, cols]
			
	return pdf


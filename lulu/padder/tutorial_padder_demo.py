import pandas as pd
from tutorial_padder import *

df = pd.read_csv('ISASimple_ICEMR_PRISM_cohort_RSRC_observations.txt', delimiter = '\t')

# Feature columns. No need to include participant id or visit dates here.

cols = ['Age at visit (years) [EUPATH_0000113]',
		'Hemoglobin (g/dL) [EUPATH_0000047]',
		'Temperature (C) [EUPATH_0000110]',
		'Height (cm) [EUPATH_0010075]',
		'Weight (kg) [EUPATH_0000732]']

# Use datetime for visit dates
df['Visit date [EUPATH_0000091]'] = pd.to_datetime(df['Visit date [EUPATH_0000091]'])
start_date = min(df['Visit date [EUPATH_0000091]']) 
end_date = max(df['Visit date [EUPATH_0000091]'])




ids = df['Participant_Id'].unique().tolist()

padded_dfs = []
for id in ids[:1]:
	padded_dfs.append(pad(df, id, cols, start_date, end_date))

pdf_head = pd.concat(padded_dfs)
pdf_head.to_csv('padded_demo__.csv', index = False)
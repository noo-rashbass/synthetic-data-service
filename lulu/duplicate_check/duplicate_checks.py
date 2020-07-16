
# A collection of functions that check for different types of duplication/ similarity between rows
# A match can suggest accidental double submission (with or without the same date)

import pandas as pd
import numpy as np
import itertools

#-------------------------------------------------------------------------------------------------
# checks for all rows, not just in instances of multiple visits
# Takes df, participant id and interested columns
# Returns: number of distinct rows that have a duplicate
def num_dup_rows(df, Participant_Id, columns = 'all'): 	
	if columns == 'all':
		columns = df.columns.tolist()

	df_id = df[df['Participant_Id']==Participant_Id].dropna(subset=['Visit date [EUPATH_0000091]'])
	df_id = df_id[columns]
	df_id_dups = df_id[df_id.duplicated()].drop_duplicates()
	num = len(df_id_dups)
	print('Number of distinct duplicated rows: ', str(num))

	if num==0:
		print('No duplicated rows for selected columns')
	else:
		print('Distinct duplicated rows: ')
		print(df_id_dups)
	return num

# -------------------------------------------------------------------------------------------------
# Gets all unordered pairs of their rows and checks which selected features match
# Horrendously slow. Not worth it if you just want counts and don't care which features/ which pairs are contributing to the counts
# Probably improved by some Iter Tools

# Returns: dataframe indexed by tuples of row pairs (could change to obvs_id pairs), containg 1 where the rows match in a particular feature.
#          series of row similarity scores indexed by tuples of row pairs.
def paired_row_dups(df, Participant_Id, columns = 'all'):
	if columns == 'all':
		columns = df.columns.tolist()

	num_features = len(columns)

	df_id = df[df['Participant_Id']==Participant_Id].dropna(subset=['Visit date [EUPATH_0000091]'])
	df_id = df_id[columns]
	print(str(len(df_id)), ' rows total')
	indices = list(range(len(df_id))) # list of all row numbers
	index_pairs = list(itertools.combinations(indices, 2)) # list of length 2 tuples
	print(str(len(index_pairs)), ' pairs of rows to check')

	index_pairs=index_pairs[:5] ### For testing

	matches = pd.DataFrame(np.zeros((len(index_pairs), num_features)))
	matches.columns = columns
	matches['row_pair'] = index_pairs


	for pair in index_pairs:
		row_a = df_id.iloc[pair[0]].tolist()
		row_b = df_id.iloc[pair[1]].tolist()
		# print(pair)
		# print(row_a)
		# print(row_b)
		pair_matches = [[1 if (row_a[i] == row_b[i]) else 0 for i in list(range(num_features))]]
		# print(pair_matches)
		matches.loc[matches.row_pair==pair, columns] = pd.DataFrame(pair_matches, columns=columns, index=[index_pairs.index(pair)]) ## Stuck here for ages not realising I had to specify an index for the row to assign

	# print(matches.head(20))
	matches.index = index_pairs
	matches = matches.drop(columns=['row_pair'])
	pair_similarity = matches.sum(axis=1).apply(lambda x: x/num_features)

	return matches, pair_similarity







######## TESTING #####################################################################
df = pd.read_csv('ISASimple_ICEMR_PRISM_cohort_RSRC_observations.txt', delimiter = '\t')
df['Visit date [EUPATH_0000091]'] = pd.to_datetime(df['Visit date [EUPATH_0000091]'])

# num_dup_rows(df, 1076)

matches, pair_similarity = paired_row_dups(df, 1076, columns = 'all')
print(matches)
print(pair_similarity)
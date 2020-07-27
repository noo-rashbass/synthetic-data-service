import pandas as pd

CATEGORICALS = [
				'Air bricks [EUPATH_0000018]',
				'Animal-drawn cart [EUPATH_0000166]',
				'Bank account [EUPATH_0000167]',
				'Bed [ENVO_00000501]',
				'Bicycle [ENVO_01000614]',
				'Boat with a motor [EUPATH_0000179]',
				'Boat without a motor [EUPATH_0000170]',
				'Car or truck [EUPATH_0000171]',
				'Cassette player [ENVO_01000578]',
				'Chair [ENVO_01000586]',
				'Clock [ENVO_01000596]',
				'Cooking fuel [EUPATH_0000023]',
				'Cupboard [ENVO_01000595]',
				'Drinking water source [ENVO_00003064]',
				'Dwelling type [ENVO_01000744]',
				'Eaves [EUPATH_0000015]',
				'Electricity [EUPATH_0021084]',
				'Floor material [EUPATH_0000006]',
				'Food problems per week [EUPATH_0000029]',
				'Household wealth index, categorical [EUPATH_0000143]',
				'Landline phone [ENVO_01000582]',
				'Lighting source [OBI_0400065]',
				'Mobile phone [ENVO_01000581]',
				'Motorcycle or scooter [ENVO_01000615]',
				'Radio [ENVO_01000577]',
				'Refrigerator [ENVO_01000583]',
				'Roof material [EUPATH_0000003]',
				'Sofa [ENVO_01000588]',
				'Sub-county in Uganda [EUPATH_0000054]',
				'Table [ENVO_01000584]',
				'Television [ENVO_01000579]',
				'Wall material [EUPATH_0000009]',
				'Watch [EUPATH_0000186]'
]

# input: DataFrame subset of categorical columns
# output: DataFrame with category options replaced with 0, 1, 2, ...
#			(Nan's will end up being assigned a number too)
# 		Series indexed by col, giving key for decoding
#		format: (<col>, <decoded cat list in order of encoding>)

def household_categorical_encoder(df_cat):
	columns = df_cat.columns.tolist()
	columns.remove('Household_Id') ## experiment doesn't mess with identifier
	# print(type(columns))
	# print(columns)

	key = pd.Series(index=columns)

	for col in columns:
		values = list(df_cat[col].unique())
		## DEALING WITH CASES
		if values == ['Yes', 'No'] or values == ['No', 'Yes']:  ## Just ensuring Y/N is sensibly replaced
			df_cat[col] = df_cat[col].apply(lambda x: 1 if x=='Yes' else 0)
			key[col] = ['No', 'Yes']
		# could add bank account case here
		else: 
			df_cat[col] = df_cat[col].apply(lambda x: values.index(x))
			key[col] = values
	return df_cat, key

# # ###### TESTING ########
# df = pd.read_csv('ISASimple_ICEMR_PRISM_cohort_RSRC_households.txt', delimiter='\t')
# df = df[['Household_Id'] + CATEGORICALS]
# print(df.head())
# df_encoded, key = household_categorical_encoder(df_cat=df)
# # df_encoded.to_csv('household_categorical_encoded.csv')
# print(df_encoded.head())
# print(key)
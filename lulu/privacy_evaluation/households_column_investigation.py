import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from bioinfokit import analys

df = pd.read_csv('ISASimple_ICEMR_PRISM_cohort_RSRC_households.txt', delimiter='\t')
# print(df.info())

# for column in df.columns:
# 	# print(df[column].nunique())
# 	print(list(df[column].unique()))

# -----------------------------------------------------------------------------------------------------------------------------------

## OVERALL DIST OF LAND SIZE ----------------------------------------------------------------------------------

# print(max(df['Acres [EUPATH_0000026]']))
# print(min(df['Acres [EUPATH_0000026]']))

# plt.hist(df['Acres [EUPATH_0000026]'], bins = 30)
# plt.xlabel('Acres of land')
# plt.ylabel('Frequency')
# plt.title('Acres of land owned by households')
# plt.savefig('Acres of land owned by households.png')

## DIST OF LAND SIZE BY WEALTH INDEX -----------------------------------------------------------------------------
# land_poorest = df[df['Household wealth index, categorical [EUPATH_0000143]']=='Poorest']['Acres [EUPATH_0000026]']
# land_middle = df[df['Household wealth index, categorical [EUPATH_0000143]']=='Middle']['Acres [EUPATH_0000026]']
# land_leastpoor = df[df['Household wealth index, categorical [EUPATH_0000143]']=='Least poor']['Acres [EUPATH_0000026]']

# plt.hist(land_poorest, histtype = 'step', range=(0,30), bins=30)
# plt.hist(land_middle, histtype = 'step', range=(0,30), bins=30)
# plt.hist(land_leastpoor, histtype = 'step', range=(0,30), bins=30)
# plt.legend(['Poorest', 'Middle', 'Least poor'])
# plt.xlabel('Acres of land')
# plt.ylabel('Frequency')
# plt.title('Household Land by Wealth Index')
# plt.savefig('Household Land by Wealth Index.png')

## IS THE DISTRIBUTION OF LAND SIZE SIGNIFICANTLY DIFFERENT BETWEEN THE TERTILES OF WEALTH? --------------------------------------------------
## get counts for chi squared test:
# df['acres_group'] = df['Acres [EUPATH_0000026]'].apply(lambda x: 'less than 1' if (x<1) else ('1 to 5' if (1<=x<=5) else 'more than 5'))
# table = pd.crosstab(df['acres_group'] ,df['Household wealth index, categorical [EUPATH_0000143]'])
# print(table)
# table.to_csv('Land_Wealth_Index.csv')

## check col totals:
# print(len(land_leastpoor))
# print(len(land_middle))
# print(len(land_poorest))

# print(analys.stat.chisq(df=table)) # 4df, p-val > .2, insignificant (just as histogram shows)
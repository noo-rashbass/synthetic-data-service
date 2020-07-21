# Lulu 21/7/20
# A fake attack on PRISM data for purposes of understanding the privacy evaluation experiment detailed in the MedGAN paper
# This in absolutely no way reflects privacy levels since we have not generated any synthetic data yet :)

# I take the publicly available PRISM household data and split into fake 'real' and fake 'synthetic'
# All synthetic data is availble to the attacker.
# A number of rows from the 'real' is compromised
# Of each compromised row, the attacker is missing values of s random attributes
# The attacker uses the 'synthetic' data to KNN model each attribute and label the missing attribute of compromised data
# The attacker is smart and will perform feature selection or weighting.
# We evaluate the attacker's success by their classification precision and classification sensitivity for different s and k values

# For the time being, also limit to purly categorical columns. Can use non KNN for numerical, or bin numericals

# Later, to actually evauate the privacy of our output, we assume the publicly available data is the 'real' data
# (ignoring the anonymisation/jittering)

# from sklearn.neighbors import KNeighborsClassifier
# classifier_attribute = KNeighborsClassifier(k)
# classifier.fit(<synthetic data>, <synthetic data labels>)

##################################################################################################################################

import pandas as pd
from household_categorical_encoder import CATEGORICALS, household_categorical_encoder
from math import factorial as fact 
from math import comb ## Python 3.8
from matplotlib import pyplot as plt


## SOME PARAMETERS OF THE EXPERIMENT --------------------------------------------
PROPORTION_REAL = 0.2
PROPORTION_COMPROMISED = 1
COLUMNS = CATEGORICALS
s = 5 # Attacker is likely to be successful if there's enough attributes remaining in each row
k = 5

## ENCODE CATEGORIES --------------------------------------------
df = pd.read_csv('ISASimple_ICEMR_PRISM_cohort_RSRC_households.txt', delimiter='\t')
df = df[['Household_Id'] + CATEGORICALS] # keeping identifier for indexing purposes
df, key = household_categorical_encoder(df_cat=df)

## GET FAKE 'REAL' AND FAKE 'SYNTHETIC' SETS --------------------------------------------
df_copy = df.copy()
df_real = df_copy.sample(frac=PROPORTION_REAL, random_state=0) # pretend real and synthetic data sets are same size
df_synth = df_copy.drop(df_real.index)

print('Number of categorical features in experiment = ', str(len(COLUMNS)))
print('Fake-Real set size = ', str(len(df_real)))
print('Fake-Synthetic set size = ', str(len(df_synth)))

## GET COMPROMISED ROWS --------------------------------------------
df_comp = df_real.sample(frac=PROPORTION_COMPROMISED, random_state=0)
print('Compromised set size = ', str(len(df_comp)))

## SIMULATE INCOMPLETE DATA -------------------------------------------
## Because my sets are so small, I'm going to pick the label attribute here and remove it from all the compromised rows. 
## Then s-1 other features will be missing to simulate attacker's difficulty in selecting columns for KNN.
## (In a the actual scenario, s random features are missing, differing from row to row. The attacker would then select the rows missing a particular feature, then KNN predict.)
## So my experiment will be evaluating privacy one column at a time. 
## Columns results can be combined to give an overall score !BUT! there's systematic error due to replacement !!!!
## It's the replacement which I'm using to increase set sizes

MISSING = 'Mobile phone [ENVO_01000581]'
# df_comp.drop([MISSING], axis=1)

# Attacker selects the following columns to predict the missing labels
PREDICTORS = ['Electricity [EUPATH_0021084]', 'Household wealth index, categorical [EUPATH_0000143]', 'Food problems per week [EUPATH_0000029]', 'Landline phone [ENVO_01000582]']

# Experimental setup says random (assume uniform) s-1 of remaining features are missing from each row 
c = len(COLUMNS)
k = len(PREDICTORS)
p = 0.5 ### INSERT FORMULA for probability a row is NOT missing any predictive features
num_rows = round(len(df_comp)*p)
print('Number of rows not missing selected predictors = ', str(num_rows))


## ATTACKER TRAINS, TESTS AND ADJUSTS THEIR MODEL ---------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# We could also conduct experiment without validating here, and just score on the compromised set
# But here I am working from the attacker's perspective
training_data, validation_data, training_labels, validation_labels = train_test_split(df_synth[PREDICTORS], df_synth[MISSING], test_size = 0.2, random_state = 1)

score=[]
for k in range(2,30):
	classifier = KNeighborsClassifier(k)
	classifier.fit(training_data, training_labels)
	score.append(classifier.score(validation_data, validation_labels))

plt.plot(list(range(2,30)), score)
plt.title('Score of KNN Mobile Phone Classifier for a range of k')
plt.xlabel('k')
plt.ylabel('Score')
plt.savefig('Mobile_KNN_scores_vs_k.png')

# Score seems to level off after k=10
print('Score on fake-Synthetic data for 10 neighbours= ', str(score[8]))

mob_classifier = KNeighborsClassifier(10)
mob_classifier.fit(training_data, training_labels)
predictions = mob_classifier.predict(df_comp[PREDICTORS])

## EVALUATE ATTACKER'S ATTEMPT USING ACTUAL VALUES NOT KNOWN TO ATTACKER -----------------------------------------
df_comp['Attacker Predicted Mobile Phone'] = pd.Series(predictions, index = df_comp.index)
df_comp[['Household_Id', MISSING, 'Attacker Predicted Mobile Phone']].to_csv('Household_compromised_mobile_predictions.csv')
print('Score for predicting missing label of fake-real data for 10 neighbours = ', str(mob_classifier.score(df_comp[PREDICTORS], df_comp[MISSING])))
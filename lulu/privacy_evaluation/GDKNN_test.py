import pandas as pd
from household_categorical_encoder import CATEGORICALS, household_categorical_encoder
from math import factorial as fact 
from math import comb ## Python 3.8
from matplotlib import pyplot as plt
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

## ENCODE CATEGORIES --------------------------------------------
df = pd.read_csv('ISASimple_ICEMR_PRISM_cohort_RSRC_households.txt', delimiter='\t')
df = df[['Household_Id'] + CATEGORICALS] # keeping identifier for indexing purposes
df, key = household_categorical_encoder(df_cat=df)

PREDICTORS = CATEGORICALS.copy()
PREDICTORS.remove('Mobile phone [ENVO_01000581]')
# print(CATEGORICALS)
# print(PREDICTORS)

## Experiment Parameters: ----------------------------------------------------------------------
p = len(PREDICTORS)
n = 10 # number of random directions to try
max_perturbation_size = 0.2 # in any particular dimension
k = 10 # Fixed throughout GD
num_iter = 100

## Get data -----------------------------------------------------------------
train_x, test_x, train_y, test_y = train_test_split(df[PREDICTORS], df['Mobile phone [ENVO_01000581]'], test_size = 0.2, random_state = 1)

######### INITIAL STEP ##########################################################################

weights = np.random.rand(p)
# print(weights)

## Get current score (already have this if it's not the first step) ---------------------------------------
classifier = KNeighborsClassifier(k, metric='wminkowski', metric_params={'w':weights})
classifier.fit(train_x, train_y)
current_score = classifier.score(test_x, test_y)
print('Iteration = ', str(0) , ' , score = ', str(current_score))
print('Starting weights = ')
print(weights)

########## ITERATE ###############################################################################

score_over_time = [current_score]

for i in range(1,num_iter):
	print('Iteration = ', str(i))

	## Get perturbed weights ----------------------------------------------------------------
	perturbations = [(np.random.rand(p)-0.5)*max_perturbation_size for i in range(n)]
	# print(perturbations)

	perturbed_weights = [np.absolute(weights+pert) for pert in perturbations]+[weights] # adding weights here to ensure we don't pick a worse weight
	perturbed_weights = [w/np.sum(w) for w in perturbed_weights] # normalising ## need to change this to something more suited to minkowski
	# print(perturbed_weights)

	## Score the perturbed weights ----------------------------------------------------------
	pert_scores = []
	for w_pert in perturbed_weights:
		classifier = KNeighborsClassifier(k, metric='wminkowski', metric_params={'w':w_pert})
		classifier.fit(train_x, train_y)
		pert_scores.append(classifier.score(test_x, test_y))

	# print('Scores of perturbed weights = ')
	# print(pert_scores)
	# print('Best score of perturbed weights = ', str(max(pert_scores)))

	new_score = max(pert_scores)
	weights = perturbed_weights[pert_scores.index(new_score)]

	# print('Score = ', str(new_score))
	# print('New weights = ')
	# print(weights)
	score_over_time.append(new_score)

########## RESULTS ##############################################################################

## Score vs iterations
plt.plot(range(num_iter), score_over_time)
plt.title('Mobile Phone GDKNN Training')
plt.xlabel('Iterations')
plt.ylabel('Score')
plt.show()

# display_weights = np.absolute(weights)
print('Final Score = ', str(new_score))
# print('Final Attribute Weights = ')
# print(weights)

## Show cols with their weighting
plt.figure(figsize=(10,8))
ax = plt.subplot()
ax.set_xticks(range(p))
ax.set_xticklabels(PREDICTORS, rotation=90)
plt.bar(range(p), weights)
plt.title('Final Attribute Weights')
plt.xlabel('Attribute')
plt.ylabel('weight')
plt.show()

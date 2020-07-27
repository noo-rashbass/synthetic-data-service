# Constants (not dependent on imputation round)

Observation ID increments by 1 every day
Participant ID and Household ID stay the same
Admitting hospital - NA (patient not actually admitted)
Age increments by 1/365 every day
Asexual plasmodium density - NA*
Asexual plasmodium present - NA*
Basis of complicated diagnosis - NA*
Complicated malaria - NA*
Diagnosis at hospitalization - NA (not actually hospitalised)

# Round 1

Abdominal pain - `Unable to assess` or `Yes` if calculated from future visit
Abdominal pain duration - 0 if `Unable to assess`, or compute from future visit duration otherwise

Anorexia - `Unable to assess` or `Yes` if calculated from future visit
Anorexia duration - 0 if `Unable to assess`, or compute from future visit duration otherwise

Cough - `Unable to assess` or `Yes` if calculated from future visit
Cough duration - 0 if `Unable to assess`, or compute from future visit duration otherwise

Days since enrollment - NA (outside range)


diarrhoea - 'unable to assess', or yes if from future visit
diarrhoea duration - 0 if unable to assess, compute from future visit otherwise
fatigue - 'unable to assess', or yes if from future visit
fatigue duration - 0 if unable to assess, compute from future visit otherwise
febrile - 'unable to assess', or yes if from future visit
febrile duration - 0 if unable to assess, compute from future visit otherwise
headache - 'unable to assess', or yes if from future visit
headache duration - 0 if unable to assess, compute from future visit otherwise
height - TODO (maybe NA or extrapolate)
haemoglobin - linerarly interpolate between values, otherwise averageing
hospital admission date - NA
hospital discharge date - NA
ITN last night - NA
jaundice - 'unable to assess', or yes if from future visit
jaundice duration - 0 if unable to assess, compute from future visit otherwise
joint pains - 'unable to assess', or yes if from future visit
joint pains duration - 0 if unable to assess, compute from future visit otherwise
malaria diagnosis - NA
Malaria diagnosis and parasite status - 'Blood smear not indicated' 
(TODO)                                  'Blood smear negative / LAMP not done'
                                        'Blood smear negative / LAMP negative'
                                        'Blood smear positive / no malaria'
                                        'Symptomatic malaria'
                                        'Blood smear negative / LAMP positive'
                                        'Blood smear indicated but not done'
                                        nan
malaria treatment - no malaria medication given
muscle aches - 'unable to assess', or yes if from future visit
muscle aches duration - 0 if unable to assess, compute from future visit otherwise
non-malaria medication - NA
other diagnosis - NA
other medical complaint - NA
Plasmodium gametocytes present - NA
seizures - 'unable to assess', or yes if from future visit
seizures duration - 0 if unable to assess, compute from future visit otherwise
severe malaria criteria - NA
subjective fever - TODO: no, yes, NA
Submicroscopic Plasmodium present - NA
temperature - interpolate linearly between values, extend backwards and forwards
visit date - incrementing by 1 every day from earliest start date to latest end date
visit type - scheduled
vomiting - 'unable to assess', or yes if from future visit
vomiting duration - 0 if unable to assess, compute from future visit otherwise
weight - linearly interpolate between values, otherwise averaging

TODOS

height - (maybe NA or extrapolate)
Malaria diagnosis and parasite status - 'Blood smear not indicated' 
(TODO)                                  'Blood smear negative / LAMP not done'
                                        'Blood smear negative / LAMP negative'
                                        'Blood smear positive / no malaria'
                                        'Symptomatic malaria'
                                        'Blood smear negative / LAMP positive'
                                        'Blood smear indicated but not done'
                                        nan
subjective fever - no, yes, NA
weight - linearly interpolate between values, but what before and after?

* we can't accurately guess without testing 
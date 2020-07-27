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
Hospital admission date - NA (not actually admitted)
Hospital discharge date - NA (not actually discharged)
ITN last night - NA*
Malaria diagnosis - NA*
Malaria diagnosis and parasite status - `Blood smear not indicated` (i.e. not tested)
Malaria treatment - `No malaria medication given` (as the patient did not visit the hospital)
Non-malaria medication - NA*
Other diagnosis - NA*
Other medical complaint - NA*
Plasmodium gametocytes present - NA*
Severe malaria criteria - NA*
Submicroscopic plasmodium present - NA*
Subjective fever - NA*
Visit date increments by 1 every day
Visit type - `Scheduled visit` (judgement call)
real - 999 (999 means fake, 1000 means real: 1000 is set for all real visits)

*we can't accurately guess without testing

NB: Weight is measured to the nearest 0.5kg

# ================================== Round 1 =======================================================

Abdominal pain - `Unable to assess` or `Yes` if calculated from future visit
Abdominal pain duration - NA if `Unable to assess`, or compute from future visit duration otherwise

Anorexia - `Unable to assess` or `Yes` if calculated from future visit
Anorexia duration - NA if `Unable to assess`, or compute from future visit duration otherwise

Cough - `Unable to assess` or `Yes` if calculated from future visit
Cough duration - NA if `Unable to assess`, or compute from future visit duration otherwise

Days since enrollment - NA (outside range of real visits)

Diarrhoea - `Unable to assess` or `Yes` if calculated from future visit
Diarrhoea duration - NA if `Unable to assess`, or compute from future visit duration otherwise

Fatigue - `Unable to assess` or `Yes` if calculated from future visit
Fatigue duration - NA if `Unable to assess`, or compute from future visit duration otherwise

Febrile - `Unable to assess` or `Yes` if calculated from future visit
Febrile duration - NA if `Unable to assess`, or compute from future visit duration otherwise

Headache - `Unable to assess` or `Yes` if calculated from future visit
Headache duration - NA if `Unable to assess`, or compute from future visit duration otherwise

Height - if the patient has the same height throughout their real visits, then use this height, otherwise NA (because same height throughout means they're adults with constant height, if not then they're a child who's height is too difficult to estimate)

Haemoglobin - average of the haemoglobin values measured at real visits

Jaundice - `Unable to assess` or `Yes` if calculated from future visit
Jaundice duration - NA if `Unable to assess`, or compute from future visit duration otherwise

Joint pains - `Unable to assess` or `Yes` if calculated from future visit
Joint pains duration - NA if `Unable to assess`, or compute from future visit duration otherwise

Muscle aches - `Unable to assess` or `Yes` if calculated from future visit
Muscle aches duration - NA if `Unable to assess`, or compute from future visit duration otherwise

Seizures - `Unable to assess` or `Yes` if calculated from future visit
Seizures duration - NA if `Unable to assess`, or compute from future visit duration otherwise

Temperature - average of temperatures measured at real visits

Vomiting - `Unable to assess` or `Yes` if calculated from future visit
Vomiting duration - NA if `Unable to assess`, or compute from future visit duration otherwise

Weight - average of weights measured at real visits if the patient is older than 20, otherwise NA (because if patient is younger than 20 then they're a child whose weight is difficult to estimate)

# ======================================== Round 2 =========================================

Abdominal pain - `Unable to assess` or `Yes` if calculated from future visit
Abdominal pain duration - NA if `Unable to assess`, or compute from future visit duration otherwise

Anorexia - `Unable to assess` or `Yes` if calculated from future visit
Anorexia duration - NA if `Unable to assess`, or compute from future visit duration otherwise

Cough - `Unable to assess` or `Yes` if calculated from future visit
Cough duration - NA if `Unable to assess`, or compute from future visit duration otherwise

Days since enrollment - calcuate based on first visit date

Diarrhoea - `Unable to assess` or `Yes` if calculated from future visit
Diarrhoea duration - NA if `Unable to assess`, or compute from future visit duration otherwise

Fatigue - `Unable to assess` or `Yes` if calculated from future visit
Fatigue duration - NA if `Unable to assess`, or compute from future visit duration otherwise

Febrile - `Unable to assess` or `Yes` if calculated from future visit
Febrile duration - NA if `Unable to assess`, or compute from future visit duration otherwise

Headache - `Unable to assess` or `Yes` if calculated from future visit
Headache duration - NA if `Unable to assess`, or compute from future visit duration otherwise

Height - linearly model the increase between each pair of visits (if one is NA then use average)
Haemoglobin - linearly model the change between each pair of visits (if one is NA then use average)

Jaundice - `Unable to assess` or `Yes` if calculated from future visit
Jaundice duration - NA if `Unable to assess`, or compute from future visit duration otherwise

Joint pains - `Unable to assess` or `Yes` if calculated from future visit
Joint pains duration - NA if `Unable to assess`, or compute from future visit duration otherwise

Muscle aches - `Unable to assess` or `Yes` if calculated from future visit
Muscle aches duration - NA if `Unable to assess`, or compute from future visit duration otherwise

Seizures - `Unable to assess` or `Yes` if calculated from future visit
Seizures duration - NA if `Unable to assess`, or compute from future visit duration otherwise

Temperature - linearly model the change between each pair of visits (if one is NA then use average)

Vomiting - `Unable to assess` or `Yes` if calculated from future visit
Vomiting duration - NA if `Unable to assess`, or compute from future visit duration otherwise

Weight - linearly model the change between each pair of visits (if one is NA then use average)

# ===================================== Round 3 ===================================================

Abdominal pain - `Unable to assess` (no future visits to infer from)
Abdominal pain duration - NA

Anorexia - `Unable to assess` (no future visits to infer from)
Anorexia duration - NA

Cough - `Unable to assess` (no future visits to infer from)
Cough duration - NA

Days since enrollment - NA (outside range of real visits)

Diarrhoea - `Unable to assess` (no future visits to infer from)
Diarrhoea duration - NA

Fatigue - `Unable to assess` (no future visits to infer from)
Fatigue duration - NA

Febrile - `Unable to assess` (no future visits to infer from)
Febrile duration - NA

Headache - `Unable to assess` (no future visits to infer from)
Headache duration - NA

Height - if the patient has the same height throughout their real visits, then use this height, otherwise NA (because same height throughout means they're adults with constant height, if not then they're a child who's height is too difficult to estimate)

Haemoglobin - average of the haemoglobin values measured at real visits

Jaundice - `Unable to assess` (no future visits to infer from)
Jaundice duration - NA

Joint pains - `Unable to assess` (no future visits to infer from)
Joint pains duration - NA

Muscle aches - `Unable to assess` (no future visits to infer from)
Muscle aches duration - NA

Seizures - `Unable to assess` (no future visits to infer from)
Seizures duration - NA

Temperature - average of temperatures measured at real visits

Vomiting - `Unable to assess` (no future visits to infer from)
Vomiting duration - NA

Weight - average of weights measured at real visits if the patient is older than 20, otherwise NA (because if patient is younger than 20 then they're a child whose weight is difficult to estimate)

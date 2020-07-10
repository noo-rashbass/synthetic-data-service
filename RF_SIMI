###replace directories for SIMI_SACT_TUMOUR for synthetic and real data###

import pandas as pd
import numpy as np

##import SIM I data and real data
SIMI_SACT_TUMOUR=pd.read_table("/Users/rebecafiadeiro/Documents/HDI/simulacrum_release_v1/sim_sact_tumour.csv", sep=",")
real_data=pd.read_table("/Users/rebecafiadeiro/Documents/HDI/simulacrum_release_v1/sim_sact_tumour.csv", sep=",")

##new table with counts of unique data
SIMI_SACT_TUMOUR_unique =pd.DataFrame({'Item': ['PRIMARY_DIAGNOSIS', 'CONSULTANT_SPECIALITY_CODE','MORPHOLOGY_CLEAN'],
                      'Syhtnetic_data': [SIMI_SACT_TUMOUR['PRIMARY_DIAGNOSIS'].nunique(), SIMI_SACT_TUMOUR['CONSULTANT_SPECIALITY_CODE'].nunique(),SIMI_SACT_TUMOUR['MORPHOLOGY_CLEAN'].nunique()],
                      'real_data': [real_data['PRIMARY_DIAGNOSIS'].nunique(), real_data['CONSULTANT_SPECIALITY_CODE'].nunique(), real_data['MORPHOLOGY_CLEAN'].nunique()]
                     })
print(SIMI_SACT_TUMOUR_unique)

##make frequency tables for SIM I and real data
SIMI_SACT_TUMOUR_CONSULTANT_SPECIALITY_CODE_FREQ=pd.value_counts(SIMI_SACT_TUMOUR['CONSULTANT_SPECIALITY_CODE']).to_frame().reset_index()
SIMI_SACT_TUMOUR_CONSULTANT_SPECIALITY_CODE_FREQ.columns=['CONSULTANT_SPECIALITY_CODE','freq_synth']

real_data_CONSULTANT_SPECIALITY_CODE_FREQ=pd.value_counts(real_data['CONSULTANT_SPECIALITY_CODE']).to_frame().reset_index()
real_data_CONSULTANT_SPECIALITY_CODE_FREQ.columns=['CONSULTANT_SPECIALITY_CODE','freq_real']

SIMI_SACT_TUMOUR_PRIMARY_DIAGNOSIS_FREQ=pd.value_counts(SIMI_SACT_TUMOUR['PRIMARY_DIAGNOSIS']).to_frame().reset_index()
SIMI_SACT_TUMOUR_PRIMARY_DIAGNOSIS_FREQ.columns=['PRIMARY_DIAGNOSIS','freq_synth']

#real_data_PRIMARY_DIAGNOSIS_FREQ=SIMI_SACT_TUMOUR_PRIMARY_DIAGNOSIS_FREQ[['PRIMARY_DIAGNOSIS']]
#real_data_PRIMARY_DIAGNOSIS_FREQ['freq_real'] =np.random.randint(1,44219, size=len(real_data_PRIMARY_DIAGNOSIS_FREQ))

real_data_PRIMARY_DIAGNOSIS_FREQ=pd.value_counts(real_data['PRIMARY_DIAGNOSIS']).to_frame().reset_index()
real_data_PRIMARY_DIAGNOSIS_FREQ.columns=['PRIMARY_DIAGNOSIS','freq_real']

SIMI_SACT_TUMOUR_MORPHOLOGY_CLEAN_FREQ=pd.value_counts(SIMI_SACT_TUMOUR['MORPHOLOGY_CLEAN']).to_frame().reset_index()
SIMI_SACT_TUMOUR_MORPHOLOGY_CLEAN_FREQ.columns=['MORPHOLOGY_CLEAN','freq_synth']

real_data_MORPHOLOGY_CLEAN_FREQ=pd.value_counts(real_data['MORPHOLOGY_CLEAN']).to_frame().reset_index()
real_data_MORPHOLOGY_CLEAN_FREQ.columns=['MORPHOLOGY_CLEAN','freq_real']

##merging the frequency tables
CONSULTANT_SPECIALITY_CODE_FREQ_merge=pd.merge(SIMI_SACT_TUMOUR_CONSULTANT_SPECIALITY_CODE_FREQ, real_data_CONSULTANT_SPECIALITY_CODE_FREQ, on='CONSULTANT_SPECIALITY_CODE',how='outer')
CONSULTANT_SPECIALITY_CODE_FREQ_merge.sort_values(by=['CONSULTANT_SPECIALITY_CODE'])
print(CONSULTANT_SPECIALITY_CODE_FREQ_merge)

PRIMARY_DIAGNOSIS_FREQ_merge=pd.merge(SIMI_SACT_TUMOUR_PRIMARY_DIAGNOSIS_FREQ, real_data_PRIMARY_DIAGNOSIS_FREQ, on='PRIMARY_DIAGNOSIS',how='outer')
PRIMARY_DIAGNOSIS_FREQ_merge.sort_values(by=['PRIMARY_DIAGNOSIS'])
print(PRIMARY_DIAGNOSIS_FREQ_merge)

MORPHOLOGY_CLEAN_FREQ_merge=pd.merge(SIMI_SACT_TUMOUR_MORPHOLOGY_CLEAN_FREQ, real_data_MORPHOLOGY_CLEAN_FREQ, on='MORPHOLOGY_CLEAN',how='outer')
MORPHOLOGY_CLEAN_FREQ_merge.sort_values(by=['MORPHOLOGY_CLEAN'])
print(MORPHOLOGY_CLEAN_FREQ_merge)

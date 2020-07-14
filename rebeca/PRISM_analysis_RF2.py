import pandas as pd
import numpy as np
import plotly as py
import plotly.express as px

table2=pd.read_table("/Users/rebecafiadeiro/Documents/HDI/ISASimple_ICEMR_PRISM_cohort_RSRC_observations.txt", sep="\t")

#changing hospital dates to dtype dates
table2['Hospital admission date [EUPATH_0000319]']=pd.to_datetime(table2['Hospital admission date [EUPATH_0000319]'])
table2['Hospital discharge date [EUPATH_0000320]']=pd.to_datetime(table2['Hospital discharge date [EUPATH_0000320]'])
table2['Visit date [EUPATH_0000091]']=pd.to_datetime(table2['Visit date [EUPATH_0000091]'])

#new column with #days of hospital duration
table2['Hospital_duration'] = (table2['Hospital discharge date [EUPATH_0000320]'] - table2['Hospital admission date [EUPATH_0000319]']).dt.days

##this gives a timedelta
##table2['Hospital_duration'] = (table2['Hospital discharge date [EUPATH_0000320]'] - table2['Hospital admission date [EUPATH_0000319]'])

#New column with Temperature range
bins=[0,37,38,39,40,np.inf]
names=['<37','37-38','38-39','39-40','40+']
table2['Temperature_range']=pd.cut(table2['Temperature (C) [EUPATH_0000110]'],bins,labels=names)

#subtable for visits that resulted in hospital admission
admitted_hospital_table2 = table2[table2["Hospital admission date [EUPATH_0000319]"] .notna()]

#subtable for people that weren't admitted to hospital - healthy people?
not_admitted_hospital_table2 = table2[table2["Hospital admission date [EUPATH_0000319]"] .isna()]

#subtable for hospital admissions
diagnosis_Malaria_table2 = table2[table2['Diagnosis at hospitalization [EUPATH_0000638]'].str.contains("Malaria", na=False)]

#subtable to look at meds and hospital stay duration for malaria diagnosis
meds_duration_table2 = diagnosis_Malaria_table2[['Visit date [EUPATH_0000091]','Hospital admission date [EUPATH_0000319]',"Hospital discharge date [EUPATH_0000320]","Hospital_duration", "Malaria treatment [EUPATH_0000740]","Non-malaria medication [EUPATH_0000059]"]]
meds_duration_table2=meds_duration_table2.sort_values(by=['Hospital_duration'])

##########################
##pie charts for each symptom

def piecharts(symptom_column, symptom_duration_column, symptom):

    admitted_hospital_table2_symptom=pd.value_counts(admitted_hospital_table2[symptom_column],dropna=False).to_frame().reset_index()
    admitted_hospital_table2_symptom.columns=[symptom,'freq']
    fig_symptom_1=px.pie(admitted_hospital_table2_symptom,values='freq',names=symptom,title='{} in admitted patients'.format(symptom))

    admitted_hospital_table2_symptom_YES=admitted_hospital_table2[admitted_hospital_table2[symptom_column]=='Yes']
    admitted_hospital_table2_symptom_YES = pd.value_counts(admitted_hospital_table2_symptom_YES[symptom_duration_column], dropna=False).to_frame().reset_index()
    admitted_hospital_table2_symptom_YES.columns=[symptom+' duration','freq']
    fig_symptom_2=px.pie(admitted_hospital_table2_symptom_YES,values='freq',names=symptom+' duration',title='Duration of {} for admitted patients with symptom (days)'.format(symptom))

    not_admitted_hospital_table2_symptom = pd.value_counts(not_admitted_hospital_table2[symptom_column], dropna=False).to_frame().reset_index()
    not_admitted_hospital_table2_symptom.columns=[symptom,'freq']
    fig_symptom_3=px.pie(not_admitted_hospital_table2_symptom,values='freq',names=symptom,title='{} in non admitted patients?'.format(symptom))

    not_admitted_hospital_table2_symptom_YES=not_admitted_hospital_table2[not_admitted_hospital_table2[symptom_column]=='Yes']
    not_admitted_hospital_table2_symptom_YES = pd.value_counts(not_admitted_hospital_table2_symptom_YES[symptom_duration_column], dropna=False).to_frame().reset_index()
    not_admitted_hospital_table2_symptom_YES.columns=[symptom+' duration','freq']
    fig_symptom_4=px.pie(not_admitted_hospital_table2_symptom_YES,values='freq',names=symptom+' duration',title='Duration of {} in non admitted patients with symptom (days)'.format(symptom))

    fig_symptom_3.show()
    fig_symptom_1.show()
    fig_symptom_4.show()
    fig_symptom_2.show()

    return admitted_hospital_table2_symptom, admitted_hospital_table2_symptom_YES, not_admitted_hospital_table2_symptom,not_admitted_hospital_table2_symptom_YES,
    fig_symptom_1,fig_symptom_2, fig_symptom_3, fig_symptom_4


piecharts('Abdominal pain [HP_0002027]','Abdominal pain duration (days) [EUPATH_0000154]','abdominal_pain')
piecharts('Anorexia [SYMP_0000523]','Anorexia duration (days) [EUPATH_0000155]','anorexia')
piecharts('Cough [SYMP_0000614]','Cough duration (days) [EUPATH_0000156]','cough')
piecharts('Diarrhea [DOID_13250]','Diarrhea duration (days) [EUPATH_0000157]','diarrhea')
piecharts('Fatigue [SYMP_0019177]','Fatigue duration (days) [EUPATH_0000158]','fatigue')
piecharts('Febrile [EUPATH_0000097]','Fever, subjective duration (days) [EUPATH_0000164]','fever')
piecharts('Headache [HP_0002315]','Headache duration (days) [EUPATH_0000159]','headache')
piecharts('Jaundice [HP_0000952]','Jaundice duration (days) [EUPATH_0000160]','jaundice')
piecharts('Joint pains [SYMP_0000064]','Joint pains duration (days) [EUPATH_0000161]','joint_pains')
piecharts('Muscle aches [EUPATH_0000252]','Muscle aches duration (days) [EUPATH_0000162]','muscle_aches')
piecharts('Seizures [SYMP_0000124]','Seizures duration (days) [EUPATH_0000163]','seizures')
piecharts('Vomiting [HP_0002013]','Vomiting duration (days) [EUPATH_0000165]','vomiting')

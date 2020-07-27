import pandas as pd
import numpy as np
import plotly as py
#pd.options.plotting.backend = "plotly"
##import tensorflow as tf
import plotly.express as px
#import plotly.graph_objects as go

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

#joint pain
#admitted to hospital
admitted_hospital_table2_joint = pd.value_counts(admitted_hospital_table2['Joint pains [SYMP_0000064]'], dropna=False).to_frame().reset_index()
admitted_hospital_table2_joint.columns=['Joint_pain','freq']
fig_joint_1=px.pie(admitted_hospital_table2_joint,values='freq',names='Joint_pain',title='Joint pain in admitted patients?')
#fig_joint_1_url = py.plot(fig_joint_1, filename='fig_joint_1',kind='pie')
#print fig_joint_1_url
#py.offline.plot(fig_joint_1, filename='fig_joint_1.html')
#py.offline.plot(fig_joint_1, include_plotlyjs=False, output_type='div')

admitted_hospital_table2_joint_YES=admitted_hospital_table2[admitted_hospital_table2['Joint pains [SYMP_0000064]']=='Yes']
admitted_hospital_table2_joint_YES = pd.value_counts(admitted_hospital_table2_joint_YES['Joint pains duration (days) [EUPATH_0000161]'], dropna=False).to_frame().reset_index()
admitted_hospital_table2_joint_YES.columns=['Joint_duration','freq']
fig_joint_2=px.pie(admitted_hospital_table2_joint_YES,values='freq',names='Joint_duration',title='Duration of joint pain in admitted patients with joint pain (days)')

#not admitted to hospital
not_admitted_hospital_table2_joint = pd.value_counts(not_admitted_hospital_table2['Joint pains [SYMP_0000064]'], dropna=False).to_frame().reset_index()
not_admitted_hospital_table2_joint.columns=['Joint_pain','freq']
fig_joint_3=px.pie(not_admitted_hospital_table2_joint,values='freq',names='Joint_pain',title='Joint pain in non admitted patients?')

not_admitted_hospital_table2_joint_YES=not_admitted_hospital_table2[not_admitted_hospital_table2['Joint pains [SYMP_0000064]']=='Yes']
not_admitted_hospital_table2_joint_YES = pd.value_counts(not_admitted_hospital_table2_joint_YES['Joint pains duration (days) [EUPATH_0000161]'], dropna=False).to_frame().reset_index()
not_admitted_hospital_table2_joint_YES.columns=['Joint_duration','freq']
fig_joint_4=px.pie(not_admitted_hospital_table2_joint_YES,values='freq',names='Joint_duration',title='Duration of joint pain in non admitted patients with joint pain (days)')

#fig_joint_1.show()
#fig_joint_3.show()
#fig_joint_2.show()
#fig_joint_4.show()
#figures_to_html([fig_joint_1, fig_joint_2, fig_joint_3])

#seizures
admitted_hospital_table2_seizures = pd.value_counts(admitted_hospital_table2['Seizures [SYMP_0000124]'], dropna=False).to_frame().reset_index()
admitted_hospital_table2_seizures.columns=['Seizures','freq']
fig_seizures_1=px.pie(admitted_hospital_table2_seizures,values='freq',names='Seizures',title='Seizures in admitted patients?')

admitted_hospital_table2_seizures_YES=admitted_hospital_table2[admitted_hospital_table2['Seizures [SYMP_0000124]']=='Yes']
admitted_hospital_table2_seizures_YES = pd.value_counts(admitted_hospital_table2_seizures_YES['Seizures duration (days) [EUPATH_0000163]'], dropna=False).to_frame().reset_index()
admitted_hospital_table2_seizures_YES.columns=['Seizures_duration','freq']
fig_seizures_2=px.pie(admitted_hospital_table2_seizures_YES,values='freq',names='Seizures_duration',title='Duration of Seizures in admitted patients with Seizures (days)')
#not admitted to hospital
not_admitted_hospital_table2_seizures = pd.value_counts(not_admitted_hospital_table2['Seizures [SYMP_0000124]'], dropna=False).to_frame().reset_index()
not_admitted_hospital_table2_seizures.columns=['Seizures','freq']
fig_seizures_3=px.pie(not_admitted_hospital_table2_seizures,values='freq',names='Seizures',title='Seizures in non admitted patients?')

not_admitted_hospital_table2_seizures_YES=not_admitted_hospital_table2[not_admitted_hospital_table2['Seizures [SYMP_0000124]']=='Yes']
not_admitted_hospital_table2_seizures_YES = pd.value_counts(not_admitted_hospital_table2_seizures_YES['Seizures duration (days) [EUPATH_0000163]'], dropna=False).to_frame().reset_index()
not_admitted_hospital_table2_seizures_YES.columns=['Seizures_duration','freq']
fig_seizures_4=px.pie(not_admitted_hospital_table2_seizures_YES,values='freq',names='Seizures_duration',title='Duration of Seizures in non admitted patients with Seizures (days)')

#jaundice
admitted_hospital_table2_jaundice = pd.value_counts(admitted_hospital_table2['Jaundice [HP_0000952]'], dropna=False).to_frame().reset_index()
admitted_hospital_table2_jaundice.columns=['jaundice','freq']
fig_jaundice_1=px.pie(admitted_hospital_table2_jaundice,values='freq',names='jaundice',title='jaundice in admitted patients?')

admitted_hospital_table2_jaundice_YES=admitted_hospital_table2[admitted_hospital_table2['Jaundice [HP_0000952]']=='Yes']
admitted_hospital_table2_jaundice_YES = pd.value_counts(admitted_hospital_table2_jaundice_YES['Jaundice duration (days) [EUPATH_0000160]'], dropna=False).to_frame().reset_index()
admitted_hospital_table2_jaundice_YES.columns=['jaundice_duration','freq']
fig_jaundice_2=px.pie(admitted_hospital_table2_jaundice_YES,values='freq',names='jaundice_duration',title='Duration of jaundice in admitted patients with jaundice (days)')
#not admitted to hospital
not_admitted_hospital_table2_jaundice = pd.value_counts(not_admitted_hospital_table2['Jaundice [HP_0000952]'], dropna=False).to_frame().reset_index()
not_admitted_hospital_table2_jaundice.columns=['jaundice','freq']
fig_jaundice_3=px.pie(not_admitted_hospital_table2_jaundice,values='freq',names='jaundice',title='jaundice in non admitted patients?')

not_admitted_hospital_table2_jaundice_YES=not_admitted_hospital_table2[not_admitted_hospital_table2['Jaundice [HP_0000952]']=='Yes']
not_admitted_hospital_table2_jaundice_YES = pd.value_counts(not_admitted_hospital_table2_jaundice_YES['Jaundice duration (days) [EUPATH_0000160]'], dropna=False).to_frame().reset_index()
not_admitted_hospital_table2_jaundice_YES.columns=['jaundice_duration','freq']
fig_jaundice_4=px.pie(not_admitted_hospital_table2_jaundice_YES,values='freq',names='jaundice_duration',title='Duration of jaundice in non admitted patients with jaundice (days)')

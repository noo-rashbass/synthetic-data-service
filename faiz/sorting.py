import pandas as pd

data = pd.read_csv('isaFull.tsv', '\t')
data = data.sort_values(['Observation_Id']).reset_index(drop=True)

data.to_csv('sorted.csv', index=False)

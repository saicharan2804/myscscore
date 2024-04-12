from SCScore import SCScorer

import pandas as pd

model = SCScorer()
model.restore()

df = pd.read_parquet('/scratch/storage0/ed19b065/saicddp/pubchem_data_118.parquet')
    
# df = df['SMILES'].tolist()
ls= df['SMILES'].tolist()
ls_gen = ls[0:5000]
ls_train = ls[5000:10000]



print('computing')
average_score = model.get_avg_score(ls_gen)
print(average_score)


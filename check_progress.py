import glob, os
import pandas as pd

matches = glob.glob(os.path.join('data/**', "progress.csv") , recursive=True)

print('\nProgress of all found progress.csv files')

for match in matches:
    print(match , '\n')
    df = pd.read_csv(match)
    print('MAX SUBGOAL SUCC RATE Train / Test')
    print(df['train_1/subgoal_succ_rate'].max(), '\t', df['test_1/subgoal_succ_rate'].max())
    print('\nSUCC RATE Train / Test')
    print(df['train/success_rate'].max(), '\t', df['test/success_rate'].max())
    print('-----------------\n\n')

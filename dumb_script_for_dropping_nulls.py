import pandas as pd 
import time
from sklearn.model_selection import train_test_split
# jnitially you have merged training_dataset with nulls

df = pd.read_csv("training_dataset.csv", sep=";")
train, valid_and_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)
valid, test = train_test_split(valid_and_test, test_size=0.5, shuffle=False)

names = ["train_set", "valid_set", "test_set"]
splits = [train, valid, test]

for split, name in zip(splits, names):
    split = split[~split["source"].isnull()]
    split.to_csv(f'{name}.csv', index=False)




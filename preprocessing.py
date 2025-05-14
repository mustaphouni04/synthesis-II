from domain_utils import DomainProcessing
from typing import Optional
from pathlib import Path
import os
import pandas as pd
from tqdm import tqdm

def translation_pairs( 
        dataset_type: str, 
        path: str,
        Sd: Optional[int] = None,
        Qd: Optional[int] = None,
        limit: Optional[int] = None,
        name: Optional[str] = None):
   
    if dataset_type == "Dataset":
        data = {}
        dataset = DomainProcessing(path, name=name)
        df = dataset.process()
        return df 

    elif dataset_type == "TM":
        tms = os.listdir(path)
        paths = []
        for tm in tqdm(tms[:limit], desc="Looking for TMs"):
            if tm.endswith(".tmx"):
                path = Path(path)
                q = path / str(tm)
                # print(q)
                paths.append(str(q))
        try:
            dataset = DomainProcessing(paths)
            df = dataset.process()
                
            return df 
        except Exception as e:
            print(f"Error: {e}")

df = translation_pairs(dataset_type="Dataset", 
                       path="ayymen/Pontoon-Translations",
                       name="en-es")

df = df.rename_columns({"source_string":"source", "target_string":"target"})

df.to_csv("pontoon_translations.csv")
print("CSV saved succesfully!!")

df = translation_pairs(dataset_type="Dataset", 
                       path="qanastek/WMT-16-PubMed",
                       name="en-es")
data = dict()
source_sentences = []
target_sentences = []

for idx, row in enumerate(df["translation"]):
    source_sentences.append(row["en"])
    target_sentences.append(row["es"])
   
data["source"] = source_sentences
data["target"] = target_sentences

df = pd.DataFrame.from_dict(data, orient='index').transpose()

df.to_csv("pubmed.csv")
print("CSV saved succesfully!!")

df = translation_pairs(dataset_type="Dataset", 
                       path="Iker/Document-Translation-en-es")

df = df.rename_columns({"en":"source", "es":"target"})

df.to_csv("news.csv")
print("CSV saved succesfully!!")


df = translation_pairs(dataset_type="Dataset", 
                       path="Iker/Reddit-Post-Translation")

df = df.rename_columns({"en":"source", "es":"target"})

df.to_csv("reddit.csv")
print("CSV saved succesfully!!")



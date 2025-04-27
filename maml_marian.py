from transformers import MarianMTModel, MarianTokenizer
import torch as th
import learn2learn as l2l
from torch import nn, optim
from pathlib import Path
from domain_utils import DomainProcessing
from typing import Tuple, Dict, List, Union, Optional
import pandas as pd
from tqdm import tqdm
import os

device = "cuda" if th.cuda.is_available() else "cpu"

model_name = "Helsinki-NLP/opus-mt-en-es"
tokenizer = MarianTokenizer.from_pretrained(model_name)
base_model = MarianMTModel.from_pretrained(model_name).to(device)

class MarianMAMLWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
        )
        return outputs.loss

def support_query_split(df: pd.DataFrame, Sd: int, Qd:int, dataset_type: str):
    data = {}

    if "train" in df:
        df = df["train"]
    else:
        pass
    
    cols = ["sentence_en", "sentence_es", "source", "target"]
    
    if dataset_type == "Dataset":
        column_names = df.column_names if hasattr(df, 'column_names') else df.columns
        ls = [col for col in cols if col in column_names]
        if ls == []:
            en, es = df["en"], df["es"]
        else:
            en, es = df[ls[0]], df[ls[1]]
    else:
        en, es = df["en"], df["es"]
    
    support_pairs = [(eng_sample, esp_sample) for eng_sample, esp_sample in zip(en[:Sd], es[:Sd])]
    query_pairs = [(eng_sample, esp_sample) for eng_sample, esp_sample in zip(en[Sd:Qd+Sd], es[Sd:Qd+Sd])]
    data["support"] = support_pairs
    data["query"] = query_pairs
    return data

def translation_pairs( 
        dataset_type: str, 
        path: str,
        Sd: int,
        Qd: int,
        limit: Optional[int] = None) -> Dict[str,List[Tuple[str,str]]]:
   
    if dataset_type == "Dataset":
        data = {}
        dataset = DomainProcessing(path)
        df = dataset.process()
        data = support_query_split(df,Sd,Qd,dataset_type)
        return data

    elif dataset_type == "TM":
        tms = os.listdir(path)
        paths = []
        for tm in tqdm(tms[:limit], desc="Looking for TMs"):
            if tm.endswith(".tmx"):
                path = Path(path)
                q = path / str(tm)
                #print(q)
                paths.append(str(q))
        try:
            dataset = DomainProcessing(paths)
            df = dataset.process()
                
            data = support_query_split(df,Sd,Qd,dataset_type)
            return data
        except Exception as e:
            print(f"Error: {e}")

tasks = []
Sd = 800
Qd = 200
webcrawl_en_es = translation_pairs(dataset_type="Dataset", 
        path="Thermostatic/parallel_corpus_webcrawl_english_spanish_1",
        Sd = Sd,
        Qd = Qd)
print(webcrawl_en_es["support"][0])
print(webcrawl_en_es["query"][0])

tasks.append(webcrawl_en_es)
financial_phrasebank = translation_pairs(dataset_type="Dataset", 
        path="NickyNicky/financial_phrasebank_traslate_En_Es",
        Sd = Sd,
        Qd = Qd)

tasks.append(financial_phrasebank)
dgt_en_es = translation_pairs(dataset_type="TM", 
        path="Vol_2012_1/",
        Sd = Sd,
        Qd = Qd,
        limit=100)
print(dgt_en_es["support"][0])
tasks.append(dgt_en_es)

automobile = translation_pairs(dataset_type="Dataset", 
        path="train_set.csv",
        Sd = Sd,
        Qd = Qd)
print(automobile["support"][0])

marianNMT = MarianMAMLWrapper(base_model)
sentence_en = "I like to play with my friends"
sentence_es = "Me gusta jugar con mis amigos"

source = tokenizer(sentence_en, return_tensors="pt", padding=True, truncate=True)
print(source)

target = tokenizer(sentence_es, return_tensors="pt", padding=True, truncate=True)
print(target)

loss = marianNMT(input_ids = source["input_ids"].to(device),
                 attention_mask= source["attention_mask"].to(device),
                 labels = target["input_ids"].to(device))

print(loss)


maml = l2l.algorithms.MAML(marianNMT, lr=1e-4, first_order=False)

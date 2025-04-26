from transformers import MarianMTModel, MarianTokenizer
import torch as th
import learn2learn as l2l
from torch import nn, optim
from pathlib import Path
from domain_utils import DomainProcessing
from typing import Tuple, Dict, List, Union, Optional
import pandas as pd

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

def support_query_split(df: pd.DataFrame, Sd: int, Qd:int):
    en, es = df["en"].values, df["es"].values
    support_pairs = [(eng_sample, esp_sample) for eng_sample, es_sample in zip(en[:Sd], es[:Sd])]
    query_pairs = [(eng_sample, esp_sample) for eng_sample, es_sample in zip(en[Sd:Qd], es[Sd:Qd])]
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
        try:
            df = df["train"]
            df = df.rename(columns={"sentence_en":"en", "sentence_es":"es"})
        except KeyError:
            pass 
        data = support_query_split(df,Sd,Qd)
        return data

    elif dataset_type == "TM":
        tms = os.listdir(path)
        paths = []
        for tm in tqdm(tms[:limit], desc="Looking for TMs"):
            if tm.endswith(".tmx"):
                path = Path(path / str(tm))
                paths.append(path)
        try:
            dataset = DomainProcessing(paths)
            df = dataset.process()
                
            data = support_query_split(df,Sd,Qd)
            return data
        except Exception as e:
            print(f"Error: {e}")

tasks = []
Sd = 800
Qd = 200
auto_finance = translation_pairs(dataset_type="Dataset", 
        path="bstraehle/en-to-es-auto-finance",
        Sd = Sd,
        Qd = Qd)
print(auto_finance["support"])

tasks.append(auto_finance)
financial_phrasebank = translation_pairs(dataset_type="Dataset", 
        path="NickyNicky/financial_phrasebank_traslate_En_Es",
        Sd = Sd,
        Qd = Qd)

tasks.append(financial_phrasebank)
dgt_en_es = translation_pairs(dataset_type="Dataset", 
        path="Vol_2012_1/",
        Sd = Sd,
        Qd = Qd,
        limit=100)
tasks.append(dgt_en_es)

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

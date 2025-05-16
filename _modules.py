from datasets import load_dataset
from typing import List, Dict, Union, Optional, Tuple
from pathlib import Path
import lxml.etree as etree
import pandas as pd
from tqdm import tqdm
import numpy as np 
from numpy import dot
from numpy.linalg import norm 
from sentence_transformers import SentenceTransformer 
from huggingface_hub import HfApi
import os
from sklearn.model_selection import train_test_split
from transformers import MarianMTModel, MarianTokenizer
import torch as th
import learn2learn as l2l
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import wandb
from dataclasses import dataclass

EURO_VOC_DOMAINS = [
    "STATISTICS",
    "ENERGY",
    "POLITICS",
    "INTERNATIONAL RELATIONS",
    "LAW",
    "ECONOMICS",
    "TRADE",
    "FINANCE",
    "SOCIAL QUESTIONS",
    "EDUCATION AND COMMUNICATIONS",
    "SCIENCE",
    "BUSINESS AND COMPETITION",
    "EMPLOYMENT AND WORKING CONDITIONS",
    "TRANSPORT",
    "ENVIRONMENT"
]

class DomainProcessing:
    def __init__(self, paths: Union[str,Path, List[Union[str, Path]]], name: Optional[str] = None):
        self.paths = [Path(p) for p in (paths if isinstance(paths,list) else [paths])]
        self.domain = None
        self.sentences = []
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.domain_embeddings = self.model.encode(EURO_VOC_DOMAINS, normalize_embeddings = True)

        if str(self.paths[0]).endswith(".csv") and isinstance(str(self.paths[0]), str):
            self.domain = pd.read_csv(self.paths[0])
        else:
            for path in self.paths:
                try:
                    self.domain = load_dataset(str(path), name=name, split="train")
                    self.content = None
                except:
                    if type(path) == Path:
                        path = str(path)
                    else:
                        try:
                            # Load a TMX file
                            tmx_file: etree._ElementTree = etree.parse(
                                    str(path), etree.XMLParser(encoding="utf-16le")
                            )
                            tmx_root: etree._Element = tmx_file.getroot()
                            self.sentences.extend(self.extract_translations(tmx_root))
                        except etree.XMLSyntaxError as e:
                            raise ValueError(f"Failed to parse TMX file: {e}")
                        except Exception as e:
                            raise RuntimeError(f"Unexpected error while processing TMX file: {e}")

    def extract_translations(self, root: etree._Element) -> List[Dict[str,str]]:
        translations = []

        for tu in tqdm(root.findall(".//tu")):
            en_sentence = None
            es_sentence = None

            for tuv in tu.findall(".//tuv"):
                lang = tuv.get("lang")
                seg = tuv.find("seg")

                if seg is not None and lang is not None:
                    if lang.startswith("EN-GB"):
                        en_sentence = seg.text.strip() if seg.text else ""
                    elif lang.startswith("ES-ES"):
                        es_sentence = seg.text.strip() if seg.text else ""

            segment_embedding = self.model.encode(en_sentence, normalize_embeddings=True)
            similarities = self.domain_embeddings @ segment_embedding.T
            best_index = np.argmax(similarities)
            best_domain = EURO_VOC_DOMAINS[best_index]

            if en_sentence and es_sentence:
                translations.append({"source":en_sentence, "target":es_sentence, "domain": best_domain})
        return translations

    @property
    def type(self):
        if self.domain is not None:
            return type(self.domain)
        else:
            return type(self.sentences)
    
    def process(self):
        if self.sentences:
            return pd.DataFrame(self.sentences)
        return self.domain

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

@dataclass
class DecoderFeatures:
    hidden_states: tuple[th.Tensor]

@dataclass
class EncoderFeatures:
    hidden_states: tuple[th.Tensor]
    encoder_last_hidden_state: th.Tensor

@dataclass
class MarianFeatures:
    encoder: EncoderFeatures
    decoder: DecoderFeatures


class MarianMAMLFeatures(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.freeze_model()

    def freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False

    @th.inference_mode()
    def forward(self, 
                input_ids, 
                attention_mask, 
                decoder_input_ids) -> MarianFeatures:
        outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
                return_dict=True
                )
        return MarianFeatures(
            encoder=EncoderFeatures(outputs.encoder_hidden_states, outputs.encoder_last_hidden_state),
            decoder=DecoderFeatures(outputs.decoder_hidden_states)
            )

def describe_features(features: MarianFeatures | EncoderFeatures | DecoderFeatures):
    match features:
        case EncoderFeatures(hidden_states=hs, encoder_last_hidden_state=last):
            return f"Encoder has {len(hs)} layers and last hidden state has shape {last.shape}."
        case DecoderFeatures(hidden_states=hs, encoder_last_hidden_state=last):
            return f"Decoder final state shape: {last.shape}"
        case MarianFeatures(encoder=enc, decoder=dec):
            return (
                f"Encoder last hidden state shape: {enc.encoder_last_hidden_state.shape}, "
                f"Decoder layers: {len(dec.hidden_states)}"
                )
        case _:
            return "Unrecognized feature structure."

def support_query_split(df: pd.DataFrame, Sd: int, Qd:int, dataset_type: str):
    data = {}

    if "train" in df:
        df = df["train"]
    else:
        pass 
    
    cols = ["sentence_en", "sentence_es", "en", "es", "source_sentence", "target_sentence"]
    
    if dataset_type == "Dataset":
        column_names = df.column_names if hasattr(df, 'column_names') else df.columns
        ls = [col for col in cols if col in column_names]
        if ls == []:
            en, es = df["source"], df["target"]
        else:
            en, es = df[ls[0]], df[ls[1]]
    else:
        en, es = df["source"], df["source"]
    
    support_pairs = [(eng_sample, esp_sample) for eng_sample, esp_sample in zip(en[:Sd], es[:Sd])]
    query_pairs = [(eng_sample, esp_sample) for eng_sample, esp_sample in zip(en[Sd:Qd+Sd], es[Sd:Qd+Sd])]
    data["support"] = support_pairs
    data["query"] = query_pairs
    return data

def dataset2df( 
        dataset_type: str, 
        path: str,
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

def feed_dicts(path: str,
               file: str):

    dict_general = {}
    if file.endswith(".en"):
        with open(path+file, "r") as src:
            return {"source":[sent.strip("\n") for sent in src.readlines()]}
                    
    elif file.endswith(".es"):
        with open(path+file, "r") as tgt:
            return {"target":[sent.strip("\n") for sent in tgt.readlines()]}

    return {}

def load_from_opus(paths: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dict1 = {}
    dict2 = {}
    for idx, path in enumerate(paths):
        file_list = os.listdir(path)
        for file in file_list:
            d = feed_dicts(path, file)
            if idx == 0:
                dict1.update(d)
            else:
                dict2.update(d)
    df1 = pd.DataFrame.from_dict(dict1, orient='index').transpose()
    df2 = pd.DataFrame.from_dict(dict2, orient='index').transpose()

    return df1, df2

def tokenize(pairs: List[Tuple[str,str]]) -> Tuple[Dict[str, th.Tensor], Dict[str, th.Tensor]]:
    source_texts = [en for en,es in pairs]
    target_texts = [es for en,es in pairs]

    inputs = tokenizer(source_texts, text_target=target_texts, return_tensors="pt", padding=True, truncation=True)

    return inputs

def create_loader(inputs,
                  labels, 
                  batch_size):

    dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"], labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def split_domain_datasets(path:str, 
                          out: str):

    for dom in domains:
        df = pd.read_csv(os.path.join(path,dom), sep=",")
        train, valid_and_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)
        valid, test = train_test_split(valid_and_test, test_size=0.5, shuffle=False)
        
        names = ["train_set", "valid_set", "test_set"]
        splits = [train, valid, test]
        file = dom.split(".")[0]

        try:
            os.mkdir(out)
        except FileExistsError:
            continue
        
        for split, name in zip(splits, names):
            split = split[~split["source"].isnull()]
            split.to_csv(f'{out}/{file}_{name}.csv', index=False)
        
    print("Splitted datasets succesfully!")

def push_to_hub(repo_name: str, model_dir: str):
    api = HfApi()
    model = model_dir
    model_files = os.listdir(model)
    
    for file in model_files:
        if ".git" not in file:
            api.upload_file(
                path_or_fileobj=os.path.join(model,file),
                path_in_repo=file.split("/")[-1],
                repo_id=repo_name,
                repo_type="model",
            )
    return 0


from datasets import load_dataset
from typing import List, Dict, Union, Optional, Tuple
from pathlib import Path
import lxml.etree as etree
import pandas as pd
from tqdm import tqdm
import numpy as np 
import random
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
import torch.nn.functional as F
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

class MarianLogits(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, 
                input_ids, 
                attention_mask, 
                decoder_input_ids):
        outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                return_dict=True
        )
        return outputs.logits

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


# --- 1) Load & freeze experts once ---
def load_and_freeze_experts(models_path, device):
    model_dirs = sorted(os.listdir(models_path))
    experts = []
    for model_dir in model_dirs:
        path = os.path.join(models_path, model_dir)
        model     = MarianMTModel.from_pretrained(path).to(device).eval()
        tokenizer = MarianTokenizer.from_pretrained(path)
        feat_ext  = MarianMAMLFeatures(model)
        teacher   = MarianLogits(model) #  (batch_size, sequence_length, config.vocab_size)) 
        experts.append((model, tokenizer, feat_ext, teacher))
    num_experts = len(experts)
    return experts, num_experts

def load_student(student_model_path, device):
    model = MarianMTModel.from_pretrained(student_model_path).to(device)
    tokenizer = MarianTokenizer.from_pretrained(student_model_path)
    feat_ext  = MarianMAMLFeatures(model)
    student   = MarianLogits(model) #  (batch_size, sequence_length, config.vocab_size)) 

    return (model, tokenizer, feat_ext, student)

def load_and_split_data(splits):
    # --- 2) Load & split data 80/20 ---
    datasets_train, datasets_test = {}, {}
    for domain, csv_path in splits.items():
        df = pd.read_csv(csv_path).dropna(subset=["source","target"])
        pairs = list(zip(df.source, df.target))
        random.shuffle(pairs)
        split = int(0.8 * len(pairs))
        datasets_train[domain] = pairs[:split]
        datasets_test[domain]  = pairs[split:]
    domains = list(datasets_train.keys())
    # Build a fixed test_samples list covering the entire held‐out set
    test_samples = []
    for idx, domain in enumerate(domains):
        for src, _ in datasets_test[domain]:
            test_samples.append((src, idx))
    
    return datasets_train, datasets_test, domains, test_samples

def build_samples(dsets, k_per_domain, domains):
    samples = []
    for idx, domain in enumerate(domains):
        pool = dsets[domain]
        picks = random.sample(pool, k=min(k_per_domain, len(pool)))
        samples += [(src, idx) for src, _ in picks]
    random.shuffle(samples)
    return samples

def chunk(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]


# infer hidden_dim
def infer_hidden_dim(experts, batch_size, device):
    _, tok0, fe0, _ = experts[0]
    enc = tok0(["Test"]*batch_size, return_tensors="pt",
               padding=True, truncation=True, max_length=512).to(device)
    dec = tok0(["<pad>"]*batch_size, return_tensors="pt",
               padding=True, truncation=True, max_length=512).to(device)
    out0 = fe0(input_ids=enc.input_ids,
               attention_mask=enc.attention_mask,
               decoder_input_ids=dec.input_ids)
    hid = out0.encoder.encoder_last_hidden_state.size(-1)

    return hid 

def train_epoch_aggregator(aggregator,
                           experts,
                           datasets_train,
                           domains,
                           batch_size,
                           optimizer,
                           criterion,
                           device,
                           epoch,
                           student_feat_ext,
                           student_logits,
                           student_tokenizer,
                           distill_config):
    
    T       = distill_config["temperature"]
    alpha   = distill_config["alpha"]
    use_rep = distill_config.get("use_rep", False)
    beta    = distill_config.get("rep_beta", 0.1)

    aggregator.train()
    total_agg_loss = 0.0 
    total_distill_loss = 0.0

    train_samples = build_samples(datasets_train, batch_size, domains)

    for batch in tqdm(chunk(train_samples, batch_size), desc=f"Epoch {epoch} Train"):
        srcs, labels = zip(*batch)
        feat_list = []

        # Extract features from each frozen expert
        for (_, tok, feat_ext, _) in experts:
            enc = tok(list(srcs),
                      return_tensors="pt",
                      padding=True, truncation=True, max_length=512
                      ).to(device)
            dec = tok(["<pad>"] * len(srcs),
                      return_tensors="pt",
                      padding="max_length", truncation=False, 
                      max_length=512
                      ).to(device)

            feats = feat_ext(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                decoder_input_ids=dec.input_ids
            )
            feat_list.append(feats.encoder.encoder_last_hidden_state)

        expert_tensor = th.stack(feat_list, dim=1)  # (B, N, S, H)
        agg_logits    = aggregator(expert_tensor)     # (B, N)
        labels_t      = th.tensor(labels, device=device)
        agg_loss      = criterion(agg_logits, labels_t)
        
        preds = agg_logits.argmax(dim=1).tolist()

        teacher_logits = []
        for i, expert_idx in enumerate(preds):
            _, tok, _, teacher = experts[expert_idx]
            enc_s = tok([srcs[i]],
                        return_tensors="pt",
                        padding=True, truncation=True, max_length=512
                        ).to(device)

            dec_s = tok(["<pad>"],
                        return_tensors="pt",
                        padding=True, truncation=True, max_length=512
                        ).to(device)
            t_logit = teacher(
                input_ids           = enc_s.input_ids,
                attention_mask      = enc_s.attention_mask,
                decoder_input_ids   = dec_s.input_ids
            )
            teacher_logits.append(t_logit.squeeze(0))
        teacher_logits = th.stack(teacher_logits, dim=0) # (B, T, V)

        enc_stu = student_tokenizer(
            srcs, return_tensors="pt", padding=True,
            truncation=True, max_length=512
        ).to(device)

        dec_stu = student_tokenizer(
            ["<pad>"] * len(srcs), return_tensors="pt",
            padding="max_length", truncation = False,
            max_length=512
        ).to(device)

        student_out = student_logits(
            input_ids         = enc_stu.input_ids,
            attention_mask    = enc_stu.attention_mask,
            decoder_input_ids = dec_stu.input_ids
        ) # (B, T, V)

        tgt_ids = student_tokenizer(
            batch_targets, # those aren't built :yet
            return_tensors="pt", padding=True,
            truncation=True, max_length=512
        ).input_ids.to(device)

        # Logit KD loss 
        student_logp    = F.log_softmax(student_out / T, dim=-1)
        teacher_p       = F.softmax(teacher_logits / T, dim=-1)
        kd_loss         = F.kl_div(student_logp, teacher_p,
                                   reduction="batchmean") * (T * T)
        # Hard CE loss on student
        ce_loss = F.cross_entropy(
            student_out.view(-1, student_out.size(-1)),
            tgt_ids.view(-1),
            ignore_index=student_tokenizer.pad_token_id
        )

        distill_loss = alpha * kd_loss + (1 - alpha) * ce_loss

        optimizer.zero_grad()
        total_loss = agg_loss + distill_loss
        total_loss.backward()
        optimizer.step()

        total_agg_loss += agg_loss.item()
        total_distill_loss += distill_loss.item()

    avg_agg_loss = total_agg_loss / (len(train_samples) / batch_size)
    avg_distill_loss = total_distill_loss / (len(train_samples) / batch_size)
    return avg_agg_loss, avg_distill_loss

def eval_epoch_aggregator(aggregator,
                          experts,
                          test_samples,
                          domains,
                          batch_size,
                          scheduler,
                          patience,
                          device,
                          epoch):
   
    aggregator.eval()
    correct_total = 0
    total_total   = 0
    correct_by_dom = {i: 0 for i in range(len(experts))}
    total_by_dom   = {i: 0 for i in range(len(experts))}

    with th.no_grad():
        for batch in tqdm(chunk(test_samples, batch_size), desc=f"Epoch {epoch} Test"):
            srcs, labels = zip(*batch)
            feat_list = []

            for (_, tokenizer, feat_ext, _) in experts:
                enc = tokenizer(list(srcs),
                                return_tensors="pt",
                                padding=True, truncation=True, max_length=512
                              ).to(device)
                dec = tokenizer(["<pad>"] * len(srcs),
                                return_tensors="pt",
                                padding=True, truncation=False, max_length=512
                              ).to(device)
                feats = feat_ext(
                    input_ids=enc.input_ids,
                    attention_mask=enc.attention_mask,
                    decoder_input_ids=dec.input_ids
                )
                feat_list.append(feats.encoder.encoder_last_hidden_state)

            expert_tensor = th.stack(feat_list, dim=1)
            logits        = aggregator(expert_tensor)
            preds         = logits.argmax(dim=1).tolist()

            for p, l in zip(preds, labels):
                total_total     += 1
                total_by_dom[l] += 1
                if p == l:
                    correct_total   += 1
                    correct_by_dom[l] += 1

    overall_acc = correct_total / total_total if total_total else 0.0
    print(f"Epoch {epoch} ▶ Test Acc: {overall_acc*100:.2f}%")
    for idx, dom in enumerate(domains):
        acc = (correct_by_dom[idx] / total_by_dom[idx]
               if total_by_dom[idx] else 0.0)
        print(f"  {dom:12s}: {acc*100:5.2f}%  ({correct_by_dom[idx]}/{total_by_dom[idx]})")

    # scheduler & early stopping logic
    scheduler.step(overall_acc)
    improved = overall_acc > eval_epoch_aggregator.best_acc
    if improved:
        eval_epoch_aggregator.best_acc = overall_acc
        eval_epoch_aggregator.no_improve = 0
    else:
        eval_epoch_aggregator.no_improve += 1

    return overall_acc, improved

eval_epoch_aggregator.best_acc = 0.0
eval_epoch_aggregator.no_improve = 0

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


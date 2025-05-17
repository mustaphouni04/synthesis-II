import os
import random
import torch
import pandas as pd
from tqdm import tqdm
from transformers import MarianTokenizer, MarianMTModel
from _modules import MarianMAMLFeatures
import importlib
module_path = "Meta-DMoE.src.transformer"
transformer_module = importlib.import_module(module_path)
aggregator = getattr(transformer_module, "Aggregator")

# --- Setup ---
models_path = "domain_models/"
model_dirs = sorted(os.listdir(models_path))   # expert names / dirs
splits = {
    "automobile": "Splits/automobile_valid_set.csv",
    "elrc":       "Splits/elrc_valid_set.csv",
    "neulab":     "Splits/neulab_valid_set.csv",
    "pubmed":     "Splits/pubmed_valid_set.csv"
}
batch_size = 8
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1) Load each domain's data into a dict: domain -> list of (src, tgt) ---
datasets = {}
for domain, path in splits.items():
    df = pd.read_csv(path).dropna(subset=["source","target"])
    datasets[domain] = list(zip(df["source"].tolist(),
                                df["target"].tolist()))

# --- 2) Build a shuffled list of (src, expert_idx) samples ---
all_samples = []
num_per_domain = batch_size  
for expert_idx, domain in enumerate(datasets):
    samples = random.sample(datasets[domain], k=num_per_domain)
    for src, tgt in samples:
        all_samples.append((src, expert_idx))

random.shuffle(all_samples)

# --- 3) Function to chunk into batches ---
def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# --- 4) Iterate over batches, extract features, collect labels ---
for batch in chunk(all_samples, batch_size):
    batch_sources, batch_labels = zip(*batch)  # src list & expert_idx list

    # 4.a) Prepare expert feature list
    expert_features = []

    # 4.b) For each frozen expert model
    for model_dir in tqdm(model_dirs, desc="Extracting expert features", leave=False):
        model_path = os.path.join(models_path, model_dir)
        model = MarianMTModel.from_pretrained(model_path).to(device).eval()
        tokenizer = MarianTokenizer.from_pretrained(model_path)
        feat_ext = MarianMAMLFeatures(model)

        # Tokenize batch of N source sentences
        enc = tokenizer(list(batch_sources), return_tensors="pt",
                        padding=True, truncation=True).to(device)

        dec_in = tokenizer(["<pad>"]*len(batch_sources),
                           return_tensors="pt", padding=True).to(device)

        feats = feat_ext(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                decoder_input_ids=dec_in["input_ids"]
        )
        # encoder side or decoder side depending on your choice:
        out = feats.encoder.encoder_last_hidden_state  # → (B, S, H)
        expert_features.append(out)

    # 4.c) Stack into (B, N, S, H)
    expert_tensor = torch.stack(expert_features, dim=1)
    # expert_tensor.shape == (batch_size, num_experts, seq_len, hidden_dim)

    # 4.d) Prepare labels tensor
    labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
    # labels.shape == (batch_size,)

    # Now we can pass expert_tensor to Aggregator:
    #logits = aggregator(expert_tensor)  # → (B, N)
    #print(logits)
    # loss = nn.CrossEntropyLoss()(logits, labels)

    print("Batch expert_tensor:", expert_tensor.shape, "labels:", labels)
    break  # remove this break to run through all batches


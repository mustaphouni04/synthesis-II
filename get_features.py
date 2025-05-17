import os
import random
import torch
import pandas as pd
from tqdm import tqdm
from torch import nn
from transformers import MarianTokenizer, MarianMTModel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW
from _modules import MarianMAMLFeatures, MarianLogits
import importlib

# --- Import Aggregator dynamically ---
module_path = "Meta-DMoE.src.transformer"
transformer_module = importlib.import_module(module_path)
Aggregator = getattr(transformer_module, "Aggregator")

# --- Config ---
models_path  = "domain_models/"
splits       = {
    "automobile": "Splits/automobile_valid_set.csv",
    "elrc":       "Splits/elrc_valid_set.csv",
    "neulab":     "Splits/neulab_valid_set.csv",
    "pubmed":     "Splits/pubmed_valid_set.csv"
}
batch_size   = 32
max_epochs   = 100
device       = "cuda" if torch.cuda.is_available() else "cpu"
patience     = 5  # early stopping patience

# --- 1) Load & freeze experts once ---
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

def build_samples(dsets, k_per_domain):
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

# --- 3) Instantiate Aggregator + optimizer + scheduler + loss ---
# infer hidden_dim
_, tok0, fe0, log0 = experts[0]
enc = tok0(["Test"]*batch_size, return_tensors="pt",
           padding=True, truncation=True, max_length=512).to(device)
dec = tok0(["<pad>"]*batch_size, return_tensors="pt",
           padding=True, truncation=True, max_length=512).to(device)
out0 = fe0(input_ids=enc.input_ids,
           attention_mask=enc.attention_mask,
           decoder_input_ids=dec.input_ids)
hid = out0.encoder.encoder_last_hidden_state.size(-1)
vocab_size = log0(input_ids=enc.input_ids,
                  attention_mask=enc.attention_mask,
                  decoder_input_ids=dec.input_ids).size(-1)

aggregator = Aggregator(
    hidden_dim=hid,
    num_experts=num_experts,
    depth=2, heads=4, mlp_dim=hid*4, dropout=0.2
).to(device)

optimizer = AdamW(
    aggregator.parameters(),
    lr=1e-4,
    weight_decay=1e-2
)
scheduler = ReduceLROnPlateau(
    optimizer, mode="max",
    factor=0.5, patience=2
)
criterion = nn.CrossEntropyLoss()

# --- 4) Training with Early Stopping & Fixed Test Set ---
best_acc = 0.0
no_improve = 0

for epoch in range(1, max_epochs+1):
    # --- Training ---
    aggregator.train()
    train_samples = build_samples(datasets_train, batch_size)
    total_loss = 0.0

    for batch in tqdm(chunk(train_samples, batch_size), desc=f"Epoch {epoch} Train"):
        srcs, labels = zip(*batch)
        feat_list = []

        for _, tokenizer, feat_ext, _ in experts:
            enc = tokenizer(list(srcs),
                            return_tensors="pt",
                            padding=True, truncation=True, max_length=512
                          ).to(device)
            dec = tokenizer(["<pad>"]*len(srcs),
                            return_tensors="pt",
                            padding=True, truncation=True, max_length=512
                          ).to(device)

            feats = feat_ext(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                decoder_input_ids=dec.input_ids
            )
            feat_list.append(feats.encoder.encoder_last_hidden_state)

        expert_tensor = torch.stack(feat_list, dim=1)  # (B, N, S, H)
        logits        = aggregator(expert_tensor)     # (B, N)
        labels_t      = torch.tensor(labels, device=device)
        loss          = criterion(logits, labels_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / (len(train_samples)/batch_size)
    print(f"Epoch {epoch} Train Loss: {avg_train_loss:.4f}")

    # --- Evaluation on the fixed test_samples ---
    aggregator.eval()
    correct_total = 0
    total_total   = 0
    correct_by_dom = {i: 0 for i in range(num_experts)}
    total_by_dom   = {i: 0 for i in range(num_experts)}

    with torch.no_grad():
        for batch in tqdm(chunk(test_samples, batch_size), desc=f"Epoch {epoch} Test"):
            srcs, labels = zip(*batch)
            feat_list = []

            for _, tokenizer, feat_ext, teacher in experts:
                enc = tokenizer(list(srcs),
                                return_tensors="pt",
                                padding=True, truncation=True, max_length=512
                              ).to(device)
                dec = tokenizer(["<pad>"]*len(srcs),
                                return_tensors="pt",
                                padding=True, truncation=True, max_length=512
                              ).to(device)
                feats = feat_ext(
                    input_ids=enc.input_ids,
                    attention_mask=enc.attention_mask,
                    decoder_input_ids=dec.input_ids
                )
                feat_list.append(feats.encoder.encoder_last_hidden_state)

            expert_tensor = torch.stack(feat_list, dim=1)
            logits        = aggregator(expert_tensor)
            preds         = logits.argmax(dim=1).tolist()
            batch_teacher_logits = []
            for i, expert_idx in enumerate(preds):
                _, tokenizer, _, teacher = experts[expert_idx]
                enc = tokenizer([srcs[i]],
                                return_tensors="pt",
                                padding=True, truncation=True, max_length=512
                                ).to(device)
                dec = tokenizer(["<pad>"],
                                return_tensors="pt",
                                padding=True, truncation=True, max_length=512
                                ).to(device)
                t_logits = teacher(
                    input_ids       = enc.input_ids,
                    attention_mask  = enc.attention_mask,
                    decoder_input_ids = dec.input_ids
                )
                batch_teacher_logits.append(t_logits.squeeze(0))
            teacher_logits = torch.stack(batch_teacher_logits, dim=0) # (B, T, V)
            labels_t      = list(labels)

            for p, l in zip(preds, labels_t):
                total_total     += 1
                total_by_dom[l] += 1
                if p == l:
                    correct_total   += 1
                    correct_by_dom[l] += 1

    overall_acc = correct_total / total_total if total_total else 0.0
    print(f"Epoch {epoch} Test Acc: {overall_acc*100:.2f}%")

    for idx, domain in enumerate(domains):
        dom_acc = correct_by_dom[idx] / total_by_dom[idx] if total_by_dom[idx] else 0.0
        print(f"  {domain:12s}: {dom_acc*100:5.2f}%  ({correct_by_dom[idx]}/{total_by_dom[idx]})")

    # --- Scheduler & Early Stopping ---
    scheduler.step(overall_acc)
    if overall_acc > best_acc:
        best_acc = overall_acc
        no_improve = 0
        print(f"  New best model (Acc: {best_acc*100:.2f}%) saved.")
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"No improvement for {patience} epochs, stopping early.")
            break

print("Training & evaluation complete.")


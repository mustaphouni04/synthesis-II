import os
import random
import torch
import pandas as pd
from tqdm import tqdm
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW
from _modules import (
                      train_epoch_aggregator,
                      eval_epoch_aggregator,
                      load_and_freeze_experts, 
                      load_and_split_data, 
                      infer_hidden_dim, 
                      load_student,
                     )
import importlib
import yaml

# --- Import Aggregator dynamically ---
module_path = "Meta-DMoE.src.transformer"
transformer_module = importlib.import_module(module_path)
Aggregator = getattr(transformer_module, "Aggregator")

# --- Config ---
with open('config.yml', 'r') as f:
    data = yaml.load(f, Loader=yaml.SafeLoader)

models_path    = data["models_path"]
splits         = data["splits"]
batch_size     = data["batch_size"]
max_epochs     = data["max_epochs"]
device         = data["device"]
patience       = data["patience"]
student_path   = data["student_model_path"]
distill_config = data["distill"]

experts, num_experts = load_and_freeze_experts(models_path, device)
# student_logits: MarianLogits object, student_feat_ext: MarianMAMLFeatures
student_model, student_tokenizer, student_feat_ext, student_logits = load_student(student_path, device)

datasets_train, datasets_test, domains, test_samples = load_and_split_data(splits)

# --- 3) Instantiate Aggregator + optimizer + scheduler + loss ---

hid = infer_hidden_dim(experts, batch_size, device)
    
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
for epoch in range(1, max_epochs+1):
    train_agg_loss, train_distill_loss = train_epoch_aggregator(
        aggregator,
        experts,
        datasets_train,
        domains,
        batch_size,
        optimizer,
        criterion,
        device,
        epoch,
        student_logits,
        student_tokenizer,
        distill_config
    )
    print(f"Epoch {epoch} ▶ Train Agg Loss: {train_agg_loss:.4f} ▶ Train Distill Loss: {train_distill_loss:.4f}")
    """
    test_acc, improved = eval_epoch_aggregator(
        aggregator,
        experts,
        test_samples,
        domains,
        batch_size,
        scheduler,
        patience,
        device,
        epoch
    )
    if improved:
        #torch.save(aggregator.state_dict(), "best_aggregator.pt")
        print(f"  ↳ New best model saved (Acc={test_acc*100:.2f}%)")

    if eval_epoch_aggregator.no_improve >= patience:
        print("Early stopping triggered.")
        break
    """

print("All done.")


"""
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
"""

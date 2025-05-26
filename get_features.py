import os
import random
import torch
import pandas as pd
from tqdm import tqdm
import learn2learn as l2l
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW
from _modules import (
                      train_step,
                      eval_epoch_aggregator,
                      load_and_freeze_experts, 
                      load_and_split_data, 
                      infer_hidden_dim, 
                      load_student,
                      build_meta_batch,     
                      build_samples,
                      train_step_query
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
inner_lr       = float(data["inner_lr"])
meta_lr        = float(data["meta_lr"])

experts, num_experts = load_and_freeze_experts(models_path, device)
# student_logits: MarianLogits object, student_feat_ext: MarianMAMLFeatures
student_model, student_tokenizer, student_feat_ext, student_logits = load_student(student_path, device)
for p in student_model.parameters():
    p.requires_grad = True

for p in student_logits.model.parameters():
    p.requires_grad = True

datasets_train, datasets_test, domains, test_samples = load_and_split_data(splits)

# --- 3) Instantiate Aggregator + optimizer + scheduler + loss ---

hid = infer_hidden_dim(experts, batch_size, device)
    
aggregator = Aggregator(
    hidden_dim=hid,
    num_experts=num_experts,
    depth=4, heads=8, mlp_dim=hid*4, dropout=0.2
).to(device)

maml_agg = l2l.algorithms.MAML(aggregator, lr=inner_lr, first_order=True)
maml_stu = l2l.algorithms.MAML(student_logits, lr=inner_lr, first_order=True)

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

# Hyperparams:
meta_batch_size = len(domains)   # one “task” per domain
k_s = 9   # support examples per domain
k_q = 9   # query  examples per domain
inner_steps = 3

meta_optimizer = torch.optim.AdamW(
    list(aggregator.parameters()) + list(student_logits.parameters()), 
    lr=meta_lr
)
ce = nn.CrossEntropyLoss()

print("Warming up student...")
for _ in tqdm(range(1000)):  # 1k warmup steps
    batch = build_samples(datasets_train, k_s+k_q, domains)[:batch_size]
    loss = train_step_query(student_logits, student_tokenizer, batch, device)
    loss.backward()
    meta_optimizer.step()
    meta_optimizer.zero_grad()

print("Warming up Aggregator...")
agg_optim = AdamW(aggregator.parameters(), lr=1e-4)

domain_weights = torch.Tensor([
    1/0.8263,
    1/0.1037,
    1/0.7464,
    1/0.7008
]).to(device)
criterion = nn.CrossEntropyLoss(weight=domain_weights)

for warmup_epoch in tqdm(range(8)):
    total_loss = 0.0
    domains_shuffled = domains.copy()
    random.shuffle(domains_shuffled)

    for domain in domains_shuffled + ['elrc']:
        domain_idx = domains.index(domain) if domain != 'elrc' else domains.index('elrc')

        k_per_domain = 10 if domain == 'elrc' else 8
        batch = build_samples(datasets_train, k_per_domain=k_per_domain, domains=[domain])
        batch = [(src, tgt, domain_idx) for src, tgt, _ in batch]

        agg_loss, _ = train_step(
                aggregator,
                student_logits,
                student_tokenizer,
                experts,
                batch,
                distill_config,
                device,
                criterion,
                only_agg=True
        )
        agg_loss.backward()
        agg_optim.step()
        agg_optim.zero_grad()
        total_loss += agg_loss.item()
    
    print(f"Aggregator Warmup Epoch {warmup_epoch}: Loss={total_loss:.3f}")

for meta_epoch in tqdm(range(max_epochs), desc="Training..."):
    meta_optimizer.zero_grad()
    meta_tasks = build_meta_batch(datasets_train, domains, k_s, k_q)
    meta_batch_size = len(meta_tasks)

    meta_loss = 0.0
    for support, query, _ in meta_tasks:
        # 1) Clone per‐task adapters
        agg_clone = maml_agg.clone()
        stu_clone = maml_stu.clone()

        agg_clone.eval()
        stu_clone.eval()

        # 2) Inner loop: adapt on support
        for _ in tqdm(range(inner_steps), desc="Running through inner steps"):
            agg_supp_loss, distill_supp_loss = train_step(
                agg_clone, stu_clone, student_tokenizer,
                experts, support,             # list of (src,tgt,idx)
                distill_config, device, criterion,
                only_agg=False       # full distill in inner loop
            )
            inner_loss = agg_supp_loss + distill_supp_loss
            maml_agg.adapt(agg_supp_loss, allow_unused=True, allow_nograd=True)
            maml_stu.adapt(distill_supp_loss, allow_unused=True, allow_nograd=True)
    
        # 3) Outer loop: evaluate clones on query
        student_ce = train_step_query(
                stu_clone,
                student_tokenizer,
                query,               # list of (src,tgt,idx)
                device,
            )
        meta_loss += student_ce 
        # average meta‐loss over all tasks
    meta_loss = meta_loss / meta_batch_size

    # 4) Meta‐update original parameters
    meta_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(aggregator.parameters()) + list(student_logits.parameters()),
        max_norm=1.0
    )
    meta_optimizer.step()

    if meta_epoch == 14:
        test_acc, improved = eval_epoch_aggregator(
            aggregator,
            experts,
            test_samples,
            domains,
            batch_size,
            scheduler,
            patience,
            device,
            meta_epoch
        )
        if improved:
            #torch.save(aggregator.state_dict(), "best_aggregator.pt")
            print(f"  ↳ New best model saved (Acc={test_acc*100:.2f}%)")

        if eval_epoch_aggregator.no_improve >= patience:
            print("Early stopping triggered.")
            break

    print(f"Meta Epoch {meta_epoch}: Meta‐Loss {meta_loss.item():.4f}")

"""
# --- 4) Training with Early Stopping & Fixed Test Set ---
for epoch in range(1, max_epochs+1):
    train_agg_loss, train_distill_loss = train_step(
        aggregator,
        student_logits,
        student_tokenizer,
        experts,
        batch,
        distill_config,
        device,
        criterion
    )
    print(f"Epoch {epoch} ▶ Train Agg Loss: {train_agg_loss:.4f} ▶ Train Distill Loss: {train_distill_loss:.4f}")

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

print("All done.")
"""

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

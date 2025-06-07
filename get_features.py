import os
import random
import torch
import math
import pandas as pd
import higher
from tqdm import tqdm
import learn2learn as l2l
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
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

criterion = nn.CrossEntropyLoss()

# Hyperparams:
meta_batch_size = len(domains)   # one “task” per domain
k_s = 9   # support examples per domain
k_q = 9   # query  examples per domain
inner_steps = 1

meta_optimizer = torch.optim.AdamW(
    list(aggregator.parameters()) + list(student_logits.parameters()), 
    lr=meta_lr
)
ce = nn.CrossEntropyLoss()

scheduler = CosineAnnealingLR(
    meta_optimizer, 
    T_max = max_epochs,
    eta_min = 1e-6
)

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
    1/0.6263,
    1/0.1037,
    1/0.5464,
    1/0.7008
]).to(device)
criterion = nn.CrossEntropyLoss(weight=domain_weights)

for warmup_epoch in tqdm(range(20)):
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

    alpha = 0.3 + 0.4 * (1 + math.cos(math.pi * min(meta_epoch, 20)/20)) / 2
    distill_config['alpha'] = alpha

    meta_loss = 0.0
    for support, query, _ in meta_tasks:
        # 1. Clone models for task-specific adaptation
        agg_clone = type(aggregator)(hidden_dim=hid, num_experts=num_experts, 
                                depth=4, heads=8, mlp_dim=hid*4, dropout=0.2).to(device)
        agg_clone.load_state_dict(aggregator.state_dict())
    
        # Only clone student parameters (no need for full model copy)
        stu_params = {n: p.clone() for n, p in student_logits.named_parameters()}
    
        # 2. Inner loop adaptation
        for inner_step in range(inner_steps):
            # Forward pass
            agg_loss, distill_loss = train_step(
                agg_clone,
                student_logits,  # Use original but with param cloning
                student_tokenizer,
                experts,
                support,
                distill_config,
                device,
                criterion,
                only_agg=False
                )
            inner_loss = agg_loss + distill_loss
        
            # Manual gradient computation and update
            grads_agg = torch.autograd.grad(inner_loss, agg_clone.parameters(), 
                                       create_graph=True, allow_unused=True)
            student_named_params = dict(student_logits.named_parameters())
            grads_stu_raw = torch.autograd.grad(inner_loss, student_named_params.values(),
                                    create_graph=True, allow_unused=True)
            grads_stu = dict(zip(student_named_params.keys(), grads_stu_raw)) 
            # Update clone parameters
            with torch.no_grad():
                for param, grad in zip(agg_clone.parameters(), grads_agg):
                    if grad is not None:
                        param -= inner_lr * grad
                    
                # Update student parameter clones
                for n, param in stu_params.items():
                    if grads_stu is not None and grads_stu.get(n) is not None:
                        param -= inner_lr * grads_stu[n]
    
        # 3. Compute query loss using adapted parameters
        # Use cloned student parameters for query
        original_params = {}
        for n, p in student_logits.named_parameters():
            original_params[n] = p.data.clone()
            p.data.copy_(stu_params[n])
            
        query_loss = train_step_query(
                student_logits,
                student_tokenizer,
                query,
                device
        )
        
        # Restore original student parameters
        for n, p in student_logits.named_parameters():
            p.data.copy_(original_params[n])
    
        # 4. Meta-update
        meta_loss = query_loss
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(aggregator.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(student_logits.parameters(), 1.0)
        meta_optimizer.step()
        meta_optimizer.zero_grad() 
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
            meta_epoch
        )
    if improved:
        #torch.save(aggregator.state_dict(), "best_aggregator.pt")
        print(f"  ↳ New best model saved (Acc={test_acc*100:.2f}%)")

    if eval_epoch_aggregator.no_improve >= patience:
        print("Early stopping triggered.")
        break
    """

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

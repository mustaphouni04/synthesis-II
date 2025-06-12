import os
import random
import torch
import math
import pandas as pd
import higher
from tqdm import tqdm
import learn2learn as l2l
from learn2learn.algorithms.meta_sgd import meta_sgd_update

import learn2learn.algorithms.meta_sgd as _ms

def adapt_with_retain(self, loss, first_order=None, retain_graph=False):
    # Copy–pasted & extended from learn2learn/algorithms/meta_sgd.py
    if first_order is None:
        first_order = self.first_order
    second_order = not first_order
    # Compute gradients w.r.t. the *underlying* module's parameters
    gradients = torch.autograd.grad(
        loss,
        self.module.parameters(),
        create_graph=second_order,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    # In-place update of self.module using the free‐function
    meta_sgd_update(self.module, self.lrs, gradients)
    return self

_ms.MetaSGD.adapt = adapt_with_retain

from learn2learn.algorithms.meta_sgd import MetaSGD
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
                      train_step_query,
                      evaluate_student_bleu
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
#domains = ['elrc', 'neulab', 'pubmed', 'automobile']

# --- 3) Instantiate Aggregator + optimizer + scheduler + loss ---

hid = infer_hidden_dim(experts, batch_size, device)
    
aggregator = Aggregator(
    hidden_dim=hid,
    num_experts=num_experts,
    depth=4, heads=8, mlp_dim=hid*4, dropout=0.2
).to(device)

for p in aggregator.parameters():
    p.requires_grad = True

maml_agg = MetaSGD(aggregator, lr=inner_lr, first_order=True)
maml_stu = MetaSGD(student_logits, lr=inner_lr, first_order=True)

optimizer = AdamW(
    aggregator.parameters(),
    lr=1e-4,
    weight_decay=1e-2
)

criterion = nn.CrossEntropyLoss()

# Hyperparams:
meta_batch_size = len(domains)   # one "task" per domain
k_s = 6   # support examples per domain
k_q = 6   # query  examples per domain
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
student_optim = AdamW(student_logits.parameters(), lr=meta_lr)
for _ in tqdm(range(1500)):  # 1.5k warmup steps
    batch = build_samples(datasets_train, k_s+k_q, domains)[:batch_size]
    loss = train_step_query(student_logits, student_tokenizer, batch, device)
    loss.backward()
    student_optim.step()
    student_optim.zero_grad()

print("Warming up Aggregator...")
agg_optim = AdamW(aggregator.parameters(), lr=1e-4)

domain_weights = torch.Tensor([
    1/0.6263,
    1/0.1037,
    1/0.5464,
    1/0.7008
]).to(device)
criterion = nn.CrossEntropyLoss(weight=domain_weights)

for warmup_epoch in tqdm(range(1)):
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
    meta_loss_total = 0.0  # Accumulate meta-loss over tasks

    for support, query, _ in meta_tasks:
        # Clone both aggregator and student for inner-loop adaptation
        agg_clone = maml_agg.clone()
        student_clone = maml_stu.clone()

        # 1. Inner Loop Adaptation
        for inner_step in range(inner_steps):
            # Compute combined loss: aggregator + distillation
            agg_loss, distill_loss = train_step(
                agg_clone,
                student_clone,  
                student_tokenizer,
                experts,
                support,
                distill_config,
                device,
                criterion,
                only_agg=False
            )
            # Combine losses for adaptation
            inner_loss = agg_loss + distill_loss  # Both modules get gradients
            
            # Adapt both clones
            agg_clone.adapt(inner_loss, first_order=True, retain_graph=True)
            student_clone.adapt(inner_loss, first_order=True, retain_graph=False)

        # 2. Outer Loop: Query Loss 
        query_loss = train_step_query(
            student_clone,
            student_tokenizer,
            query,
            device
        )

        # Accumulate meta-loss
        meta_loss_total += query_loss + 0.1 * agg_loss

    # 3. Meta-Update: Backprop through adaptation steps
    meta_loss_total.backward()

    # 4. Gradient Monitoring & Clipping
    agg_grads = [p.grad for p in aggregator.parameters() if p.grad is not None]
    stu_grads = [p.grad for p in student_logits.parameters() if p.grad is not None]

    if agg_grads:
        agg_norm = torch.norm(torch.stack([g.norm() for g in agg_grads]))
    else:
        agg_norm = torch.tensor(0.0)
    
    if stu_grads:
        stu_norm = torch.norm(torch.stack([g.norm() for g in stu_grads]))
    else:
        stu_norm = torch.tensor(0.0)

    print(f"Grad Norms: Agg={agg_norm:.4f}, Student={stu_norm:.4f}")

    # Clip gradients to prevent explosion
    torch.nn.utils.clip_grad_norm_(aggregator.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(student_logits.parameters(), 1.0)

    # Meta-parameter update
    meta_optimizer.step()
    meta_optimizer.zero_grad()

    # --- Evaluation & Logging ---
    print(f"Meta Epoch {meta_epoch}: Meta-Loss {meta_loss_total.item():.4f}")
    
    if meta_epoch % 5 == 0:
        print("⇨ Evaluating student BLEU on all test domains …")
        overall_bleu, per_domain_bleu = evaluate_student_bleu(
            datasets_test, student_model, student_tokenizer, device
        )
        print(f"Overall student BLEU: {overall_bleu:.2f}")
        for dom, score in per_domain_bleu.items():
            print(f"  {dom:12s}: {score:.2f}")

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

    scheduler.step()

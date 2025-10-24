import os,json
import torch
import numpy as np
from monai.data import DataLoader
from BraTSDataset.BraTSDataset import BraTSDataset
from train_step import train_epochs

device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- 1.算熵 ----------
def mc_entropy(model, img, T=10):
    model.train()          # 保持 dropout
    preds = []
    with torch.no_grad():
        for _ in range(T):
            preds.append(torch.softmax(model(img), 1).cpu())
    mean_prob = torch.stack(preds).mean(0)
    entropy = -(mean_prob * torch.log(mean_prob + 1e-8)).sum(0)
    return entropy.mean().item()          # 标量

# ---------- 2. 挑top-k个不确定 ----------
def select_topk_uncertain(model, pool_loader, labeled_idx, device, k=5):
    scores = []
    for batch in pool_loader:
        case_idx = batch["case_idx"].item()
        if case_idx in labeled_idx:        # 已标注就跳过
            continue
        img = batch["image"].to(device)
        scores.append((case_idx, mc_entropy(model, img)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in scores[:k]]

# ---------- 3.主动学习 ----------
def active_learning_loop(model, train_dataset, val_dataset, device,transforms,
                         budget=50, rounds=10, epochs_per=5,ensemble_iter=0):
    """先 20 % 种子 → 每轮挑 budget 例 → 再训 epochs_per"""
    pool_indices = train_dataset.indices.copy()          # 全集
    labeled_idx = set(np.random.choice(len(pool_indices),
                                       size=int(0.2*len(pool_indices)),
                                       replace=False))
    for r in range(rounds):
        print(f"\n==== AL Round {r+1}/{rounds}  labeled={len(labeled_idx)} ====")
        # 当前已标注子集
        labeled_cases = [train_dataset.dataset.data[i] for i in labeled_idx]
        subset_ds = BraTSDataset(labeled_cases, transforms)
        subset_loader = DataLoader(subset_ds, batch_size=1, shuffle=True)
        # 训练
        model = train_epochs(model, subset_loader, val_dataset, device, epochs_per,None)
        # 挑新样本
        pool_cases = [train_dataset.dataset.data[i] for i in pool_indices]
        pool_loader = DataLoader(BraTSDataset(pool_cases, transforms), batch_size=1, shuffle=False)
        new_idx = select_topk_uncertain(model, pool_loader, labeled_idx,device,k=budget)
        labeled_idx.update(new_idx)
    return model,labeled_idx
import os

import torch.nn.functional as F
import monai
import torch
from notion_csv import save_metrics_csv
from cal_num import compute_all_metrics
data_dir = r'../data/BraTS2021_preprocess'

# 小样本快速训练
def train_epochs(model, train_loader, val_dataset, device, epochs, loss_fn=None):
    """
    通用训练壳
    :param model:       要训练的网络
    :param train_loader:      训练集 DataLoader
    :param val_dataset:  验证集 DataLoader（仅调 scheduler，可不用就传 None）
    :param epochs:      跑多少 epoch
    :param loss_fn:     外部自定义损失，None 就用默认 DiceLoss
    :return:            训练好的模型
    """
    print('start to train')
    # 1. 默认用Dice损失
    if loss_fn is None:
        loss_fn = monai.losses.DiceLoss(to_onehot_y=True, softmax=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,eta_min=1e-6)
    for epoch in range(epochs):
        model.train()
        running_loss=0.0
        for batch in train_loader:
            img = batch["image"].to(device)
            seg = batch["seg"].to(device)
            seg[seg == 4] = 3          # BraTS 标签 4→3
            optimizer.zero_grad()
            logits = model(img)
            loss = loss_fn(logits, seg)

            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
        scheduler.step()
        avg_loss=running_loss/len(train_loader)
        if (epoch+1) % 1==0:
            model.eval()
            with torch.no_grad():
                batch = next(iter(train_loader))
                img = batch["image"].to(device)
                seg = batch["seg"].to(device)
                seg[seg == 4] = 3
                pred = model(img)
                pred = F.softmax(pred, dim=1)
            dict=compute_all_metrics(pred,seg,device,100)
            save_metrics_csv("../logs/train_log.csv",epoch+1,avg_loss,dict)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    print('end train')
    return model


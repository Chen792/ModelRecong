import os

import torch.nn.functional as F
import monai
import torch
from notion_csv import save_metrics_csv
from cal_num import compute_all_metrics
data_dir = r'../data/BraTS2021_preprocess'
import gc

def force_memory_cleanup():
    """强制内存清理函数"""
    # 1. Python垃圾回收
    gc.collect()

    # 2. 清空GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # 等待所有操作完成

    # 3. 清空CPU缓存（如果有GPU）
    if hasattr(torch.cuda, 'reset_peak_memory_stats'):
        torch.cuda.reset_peak_memory_stats()
# 小样本快速训练
def train_epochs(model, train_loader, val_dataloader, device, epochs, loss_fn=None):
    """
    通用训练壳
    :param model:       要训练的网络
    :param train_loader:      训练集 DataLoader
    :param val_dataloader:  验证集 DataLoader（仅调 scheduler，可不用就传 None）
    :param epochs:      跑多少 epoch
    :param loss_fn:     外部自定义损失，None 就用默认 DiceLoss
    :return:            训练好的模型
    """
    force_memory_cleanup()
    print('start to train')
    # 1. 默认用Dice损失
    if loss_fn is None:
        loss_fn = monai.losses.DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,eta_min=1e-6)
    best_val_dice=0.0
    best_model_state=None
    os.makedirs(f'../logs',exist_ok=True)
    for epoch in range(epochs):
        best_val_loss=10000
        model.train()
        running_loss=0.0
        optimizer.zero_grad()
        for batch_idx,batch in enumerate(train_loader):
            img = batch["image"].to(device)
            seg = batch["seg"].to(device)
            seg[seg == 4] = 3          # BraTS 标签 4→3
            optimizer.zero_grad()
            logits = model(img)
            # pred=F.softmax(logits,dim=1)
            loss = loss_fn(logits, seg)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
        scheduler.step()
        avg_loss=running_loss/len(train_loader)
        model.eval()
        with torch.no_grad():
            batch = next(iter(train_loader))
            img = batch["image"].to(device)
            seg = batch["seg"].to(device)
            seg[seg == 4] = 3
            pred = model(img)
            pred = F.softmax(pred, dim=1)
        dict = compute_all_metrics(pred, seg, device, 5)
        save_metrics_csv("../logs/train_log.csv", epoch + 1, avg_loss, dict)
        if val_dataloader is not None:
            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for val_batch in val_dataloader:
                    val_img = val_batch["image"].to(device)
                    val_seg = val_batch["seg"].to(device)
                    val_seg[val_seg == 4] = 3
                    val_logits = model(val_img)
                    val_loss += loss_fn(val_logits, val_seg).item()
            with torch.no_grad():
                batch = next(iter(val_dataloader))
                img = batch["image"].to(device)
                seg = batch["seg"].to(device)
                seg[seg == 4] = 3
                pred = model(img)
                pred = F.softmax(pred, dim=1)
            dict = compute_all_metrics(pred, seg, device, 5)
            avg_val_loss = val_loss / len(val_dataloader)
            save_metrics_csv("../logs/val_log.csv", epoch + 1, avg_val_loss, dict)
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()

            # 打印训练和验证损失
            print(f"Epoch {epoch + 1}/{epochs},Val Loss: {avg_val_loss:.4f}")

        if epoch%5==0:
            force_memory_cleanup()
        # 打印平均损失
        print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}")

    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with Val loss: {best_val_loss:.4f}")
    print('end train')
    return model


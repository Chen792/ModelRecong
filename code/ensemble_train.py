from monai.networks.nets import UNet
from BraTSDataset.getData import getDatasetAndLoaderAndOthers
import torch
from monai.data import DataLoader
import torch.nn.functional as F
import numpy as np
import copy
from monai.losses import DiceLoss
from torch.nn import init
from notion_csv import save_metrics_csv
from cal_num import compute_all_metrics
import time
import gc
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
def reinit_weights(model):
    """权重重新初始化函数"""
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv3d, torch.nn.Linear)):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

def create_bootstrap_dataset(dataset, seed=42, sample_ratio=0.8):
    """为每个模型创建bootstrap采样子集"""
    torch.manual_seed(seed)
    dataset_size = len(dataset)
    sample_size = int(dataset_size * sample_ratio)

    # 有放回随机采样
    indices = torch.randint(0, dataset_size, (sample_size,))
    return torch.utils.data.Subset(dataset, indices)


def create_diverse_model():

    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        channels=[32,64,128,256,512],
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.2
    ).to(device)

    return model


def ensemble_train_model(dataset, num_models=5, base_epochs=20):
    """专门的Ensemble模型训练函数"""
    models = []
    # 获取验证数据集用于早停和模型选择
    _, val_dataset, _, _, _ = getDatasetAndLoaderAndOthers()
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # 训练每个模型
    for model_idx in range(num_models):
        start_time = time.time()

        # 1. 创建不同的数据子集
        model_dataset = create_bootstrap_dataset(dataset, seed=model_idx, sample_ratio=0.8)

        # 2. 创建数据加载器
        try:
            train_loader = DataLoader(model_dataset, batch_size=1, shuffle=True,
                                      num_workers=2, pin_memory=True)
        except RuntimeError:
            train_loader = DataLoader(model_dataset, batch_size=1, shuffle=True,
                                      num_workers=2, pin_memory=True)

        # 3. 创建有差异的模型
        model = create_diverse_model()
        model.apply(reinit_weights)
        # 4. 设置不同的训练参数
        epochs = base_epochs

        # 5. 训练模型
        trained_model = train_single_model(
            model, train_loader, val_loader, epochs
        )

        models.append(trained_model)
        # 6. 计算训练时间
        training_time = time.time() - start_time
        print(f'Training completed in {training_time:.2f} seconds')

    print('\nEnsemble training completed!')
    return models


def train_single_model(model, train_loader, val_loader, epochs):
    UNIFIED_LR = 0.0008
    UNIFIED_WEIGHT_DECAY = 0.01

    optimizer = torch.optim.AdamW(model.parameters(), lr=UNIFIED_LR, weight_decay=UNIFIED_WEIGHT_DECAY)
    print(f'Optimizer: AdamW (UNIFIED), lr={UNIFIED_LR}, weight_decay={UNIFIED_WEIGHT_DECAY}')

    # 【修改：统一使用 Cosine Annealing 学习率调度器】
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    print(f'Scheduler: CosineAnnealingLR (UNIFIED), T_max={epochs}')

    # 损失函数 (保持不变)
    criterion = DiceLoss(softmax=True, to_onehot_y=True, squared_pred=True)

    best_model_wts = copy.deepcopy(model.state_dict())

    print(f'Training for {epochs} epochs')
    best_val_loss = 10000
    for epoch in range(epochs):
        # 训练阶段
        start=time.time()
        model.train()
        epoch_train_loss = 0.0
        num_batches = 0

        for batch_data in train_loader:
            inputs = batch_data["image"].to(device)
            seg = batch_data["seg"].to(device)
            seg[seg==4]=3
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)

            # 计算损失
            loss = criterion(outputs, seg)

            # 反向传播
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            num_batches += 1
        end=time.time()
        model.eval()
        with torch.no_grad():
            batch = next(iter(train_loader))
            img = batch["image"].to(device)
            seg = batch["seg"].to(device)
            seg[seg == 4] = 3
            pred = model(img)
            pred = F.softmax(pred, dim=1)
        avg_train_loss = epoch_train_loss / num_batches
        dict = compute_all_metrics(pred, seg, device, 5)
        dict['time']=end-start
        save_metrics_csv("../logs/train_log.csv", epoch + 1, avg_train_loss, dict)
        # 验证阶段
        model.eval()
        epoch_val_loss=0.0
        val_batches = 0

        with torch.no_grad():
            for val_data in val_loader:
                val_inputs = val_data["image"].to(device)
                val_labels = val_data["seg"].to(device)
                val_labels[val_labels==4]=3
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)
                epoch_val_loss += val_loss.item()
                val_batches += 1

        avg_val_loss = epoch_val_loss / val_batches
        with torch.no_grad():
            batch = next(iter(val_loader))
            img = batch["image"].to(device)
            seg = batch["seg"].to(device)
            seg[seg == 4] = 3
            pred = model(img)
            pred = F.softmax(pred, dim=1)
        dict = compute_all_metrics(pred, seg, device, 5)
        save_metrics_csv("../logs/val_log.csv", epoch + 1, avg_val_loss, dict)

        scheduler.step()

        # 早停和模型保存
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = model.state_dict().copy()

        # 打印训练和验证损失
        print(f"Epoch {epoch + 1}/{epochs},Val Loss: {avg_val_loss:.4f}")
        print(f"Epoch {epoch + 1}/{epochs},train Loss: {avg_train_loss:.4f}")

    if epoch % 5 == 0:
        force_memory_cleanup()

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    print(f'Best validation Dice: {best_val_loss:.4f}')

    return model


def check_model_diversity(model1, model2, val_loader):
    """检查两个模型预测的多样性"""
    model1.eval()
    model2.eval()

    disagreements = 0
    total_samples = 0

    with torch.no_grad():
        for batch_data in val_loader:
            inputs = batch_data["image"].to(device)

            # 获取两个模型的预测
            pred1 = torch.argmax(model1(inputs), dim=1)
            pred2 = torch.argmax(model2(inputs), dim=1)

            # 计算不一致的像素比例
            disagreement = (pred1 != pred2).float().mean().item()
            disagreements += disagreement
            total_samples += 1

    diversity_score = disagreements / total_samples if total_samples > 0 else 0
    return diversity_score



# 使用示例
if __name__ == "__main__":
    # 获取训练数据集
    train_dataset, _, _, _, _ = getDatasetAndLoaderAndOthers()

    # 训练集成模型
    models = ensemble_train_model(train_dataset)

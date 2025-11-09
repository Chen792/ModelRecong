import torch
from monai.networks.nets import UNet
import torch.nn.functional as F
import os
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

def calculate_dice_score(model, data_loader, device):
    """
    评估模型在给定 DataLoader 上的平均 Dice Score (MONAI 实现)。

    用于计算集成模型的权重。
    """
    model.eval()
    # include_background=False 意味着只计算前景类别 (1, 2, 3) 的平均 Dice
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")

    # 假设您的 3D UNet 训练时使用了 128^3 的 patch
    roi_size = (128, 128, 128)
    sw_batch_size = 4  # 滑动窗口批次大小

    # 使用 tqdm 显示评估进度
    for batch_data in data_loader:
        inputs = batch_data["image"].to(device)
        labels = batch_data["seg"].to(device)

        # 统一标签，将类别 4 映射到类别 3
        labels[labels == 4] = 3

        with torch.no_grad():
            # 使用滑动窗口推理
            outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, model)

            # 转换为 Softmax 概率
            outputs = F.softmax(outputs, dim=1)

            # 转换为预测标签 (为了 DiceMetric 期望的 one-hot 格式，这里传入 (N, 1, D, H, W) 的 argmax 结果)
            predictions = torch.argmax(outputs, dim=1, keepdim=True)

            # 更新 DiceMetric (y_pred 期望是 one-hot 形式, 但 MONAI 的 DiceMetric 可以处理 argmax 结果)
            dice_metric(y_pred=predictions, y=labels)

    # 聚合所有批次的 Dice Score
    mean_dice = dice_metric.aggregate().item()
    dice_metric.reset()

    return mean_dice
def compute_uncertainty(probs):
    """
    计算每个模型预测的不确定性
    probs: [num_models, B, C, H, W, D]
    返回: [num_models, B] 每个模型每个样本的不确定性
    """
    num_models, batch_size = probs.shape[0], probs.shape[1]
    uncertainties = torch.zeros(num_models, batch_size, device=probs.device)

    for i in range(num_models):
        for b in range(batch_size):
            # 计算预测熵作为不确定性度量
            prob = probs[i, b]  # [C, H, W, D]
            entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=0)  # [H, W, D]
            uncertainties[i, b] = entropy.mean()  # 平均熵

    return uncertainties

#单个模型的推理
def Single_Model_pred(image,device,model,is_MC):
    """

    :param image: 输入的图像
    :param device: 在哪个设备上处理
    :param model_path: 模型的父文件夹路径
    :param mode: 哪种训练？baseline？Active Learning? 0=baseline;1=AL
    :param is_MC: 需不需要进行MC Dropout
    :return: 预测概率
    """
    #MC Dropout层
    # 2. 加载保存的参数
    if is_MC:
        model.train()
    else:
        model.eval()
    with torch.no_grad():
        preds=model(image)
    pred=F.softmax(preds,dim=1) #[B,C,H,W,D]
    prob_labels=torch.argmax(pred,dim=1) #[B,H,W,D]
    del preds
    torch.cuda.empty_cache()
    return pred,prob_labels


#集成模型的推理
def Mutil_Model_pred(image, device, models, is_MC,weight=None):
    """
    改进的集成模型预测函数

    :param image: 输入的图像 [B, C, H, W, D]
    :param device: 在哪个设备上运行
    :param models: 集成模型列表
    :param is_MC: 是否需要train模式来保证MC Dropout
    :param ensemble_method: 集成方法 'average'|'weighted'|'majority'|'uncertainty_weighted'
    :return: 预测概率和预测标签
    """
    batch_size = image.shape[0]
    # 收集所有模型的预测
    all_logits = []
    all_probs=[]
    for i, model in enumerate(models):
        # 设置模型模式
        if is_MC:
            model.train()  # MC Dropout需要train模式
        else:
            model.eval()

        with torch.no_grad():
            logits = model(image)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs)
            all_logits.append(logits)

    # 最终概率和预测标签
    all_probs = torch.stack(all_probs)
    if weight is None or len(weight) != len(models):
        # 如果没有提供权重，或者权重数量不匹配，则退化为简单平均
        print("Warning: Using Simple Averaging. Consider providing Dice-based weights.")
        ensemble_probs = all_probs.mean(dim=0)
    else:
        # **加权平均 (Weighted Averaging)**
        weights_tensor = torch.tensor(weight, dtype=all_probs.dtype).to(device)

        # 扩展权重维度以匹配 stacked_probs (用于广播)
        # [N] -> [N, 1, 1, 1, 1, 1] (Num_Models, 1, 1, 1, 1, 1)
        expanded_weights = weights_tensor.view(-1, 1, 1, 1, 1, 1)

        # 加权求和：(N, B, C, D, H, W) * (N, 1, 1, 1, 1, 1) -> (N, B, C, D, H, W)
        weighted_probs = all_probs * expanded_weights

        # 在模型维度 N 上求和，得到最终概率: (B, C, D, H, W)
        ensemble_probs = torch.sum(weighted_probs, dim=0)
    prob_labels = torch.argmax(ensemble_probs, dim=1)

    return ensemble_probs, prob_labels





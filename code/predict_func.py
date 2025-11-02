import torch
from monai.networks.nets import UNet
import torch.nn.functional as F
import os


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
def Mutil_Model_pred(image, device, models, is_MC):
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
    all_probs = []

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

    # 堆叠所有预测 [num_models, B, C, H, W, D]
    all_probs = torch.stack(all_probs)  # [5, B, 4, H, W, D]
    # 基于不确定性的加权
    uncertainties = compute_uncertainty(all_probs)  # [5, B]
    weights = 1.0 / (uncertainties + 1e-8)  # 不确定性越低，权重越高
    weights = weights / weights.sum(dim=0, keepdim=True)  # 归一化

    # 扩展权重维度以匹配概率张量
    weights = weights.view(-1, batch_size, 1, 1, 1, 1)
    weighted_probs = all_probs * weights
    ensemble_probs = weighted_probs.sum(dim=0)

    # 计算最终预测标签
    prob_labels = torch.argmax(ensemble_probs, dim=1)  # [B, H, W, D]

    # 清理内存
    del all_probs
    torch.cuda.empty_cache()

    return ensemble_probs, prob_labels





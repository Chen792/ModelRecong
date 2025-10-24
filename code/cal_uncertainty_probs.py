import torch
from predict_func import Mutil_Model_pred,Single_Model_pred
import os
# 计算概率图 (Softmax)
def cal_prob(pred):
    """

    :param pred: [B,C,H,W,D]
    :return:
    """
    os.makedirs("../prob_imgs", exist_ok=True)
    probs = pred.detach().cpu().numpy()[0]  # shape: [C, H, W, D]
    return probs #[C,H,W,D]

#计算不确定性
def cal_uncertainty_online(model,image,device,T,is_Multi):
    mean_probs = None

    for i in range(T):
        if is_Multi:
            pred,_=Mutil_Model_pred(image,device,model,True)
        else:
            pred, _ = Single_Model_pred(image, device, model, is_MC=True)
        prob = pred[0]  # [C,H,W,D]
        if mean_probs is None:
            mean_probs = prob / T
        else:
            mean_probs += prob / T
        del pred,prob
        torch.cuda.empty_cache()

    # 计算熵
    uncertainty = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=0)
    return uncertainty.detach().cpu().numpy()

import torch, numpy as np, torch.nn.functional as F
from monai.metrics import DiceMetric, HausdorffDistanceMetric
# ---------- 1. 基础分割指标 ----------
def dice_iou_hd95(pred, gt, include_background=True):
    dice_fn = DiceMetric(include_background=include_background, reduction="none")
    hd_fn   = HausdorffDistanceMetric(include_background=include_background, reduction="none")
    dice_fn(y_pred=pred, y=gt)
    hd_fn(y_pred=pred, y=gt)
    dice = dice_fn.aggregate().cpu().numpy()        # [C]
    hd   = hd_fn.aggregate().cpu().numpy()
    iou  = dice / (2 - dice + 1e-8)
    return dice, iou, hd

# ---------- 2. 校准指标 ----------
def calibrate_ece_brier(probs, target, n_bins=15):
    """ECE & Brier (bin=15)"""
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers, bin_uppers = bin_boundaries[:-1], bin_boundaries[1:]
    ece, brier = 0., 0.

    # 1. 确保 target 是正确的 label 形式 [B, H, W, D]
    target_labels = target.squeeze(1).long()  # [B, H, W, D]

    max_conf, pred_labels = probs.max(dim=1)  # pred_labels: [B, H, W, D]


    # 将所有维度展平，以便进行基于置信度的分箱和索引
    max_conf_flat = max_conf.flatten()
    pred_labels_flat = pred_labels.flatten()
    target_labels_flat = target_labels.flatten()  # [N]
    probs_flat = probs.permute(0, 2, 3, 4, 1).reshape(-1, probs.shape[1])  # [N, C]

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # 使用展平后的置信度进行分箱
        in_bin = (max_conf_flat > bin_lower) & (max_conf_flat <= bin_upper)
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin > 0:
            # ECE - 使用 label 形式进行准确率计算
            # pred_labels_flat[in_bin] 和 target_labels_flat[in_bin] 都是 [N_in_bin] 的 label
            accuracy = (pred_labels_flat[in_bin] == target_labels_flat[in_bin]).float().mean()
            confidence = max_conf_flat[in_bin].mean()
            ece += torch.abs(accuracy - confidence) * prop_in_bin

            # Brier - 需要 one-hot 形式
            # probs_flat[in_bin] 是 [N_in_bin, 4] 的概率
            # F.one_hot(...) 得到 [N_in_bin, 4] 的 one-hot
            target_one_hot_in_bin = F.one_hot(target_labels_flat[in_bin], num_classes=probs.shape[1]).float()
            brier += torch.mean((probs_flat[in_bin] - target_one_hot_in_bin).pow(2)) * prop_in_bin

    return ece.item(), brier.item()


# ---------- 3. 熵 ----------
def entropy_from_prob(prob):
    prob = F.softmax(prob, dim=1)
    return -(prob * torch.log(prob + 1e-8)).sum(1)   # [B,H,W,D]

# ---------- 4. Overlap@k / Recall@k ----------
def overlap_recall_k(uncertainty, err, k=100):
    N=len(uncertainty)
    k=N*0.05
    thresh = np.percentile(uncertainty, 100 - k / uncertainty.size * 100)
    pred_high = uncertainty > thresh
    overlap = np.logical_and(pred_high, err).sum()
    recall = overlap / (err.sum() + 1e-8)
    return overlap / (pred_high.sum() + 1e-8), recall

# ---------- 5. 一键全部指标 ----------
def compute_all_metrics(pred, seg, device, k=100):
    dice_metric = DiceMetric(include_background=True, reduction="none")
    hd_metric   = HausdorffDistanceMetric(include_background=True, reduction="none")
    nlls, eces, briers = [], [], []
    uncertainty_list, error_list = [], []
    pred=pred.to(device)
    seg=seg.to(device)
    seg[seg==4] = 3
    pred_labels=torch.argmax(pred,dim=1,keepdim=False)
    pred_onehot = F.one_hot(pred_labels, num_classes=4)
    pred_onehot = pred_onehot.permute(0, 4, 1, 2, 3).float()
    # seg 也是 [B, 1, H, W, D]
    seg=seg.long()
    seg_onehot = F.one_hot(seg.squeeze(1), num_classes=4)
    seg_onehot = seg_onehot.permute(0, 4, 1, 2, 3).float()
    # 2. 分割指标
    dice_metric(y_pred=pred_onehot, y=seg_onehot)
    hd_metric(y_pred=pred_onehot, y=seg_onehot)

    # 3. 校准
    nll = F.cross_entropy(pred, seg.squeeze(1).long())
    nlls.append(nll.item())
    ece, brier = calibrate_ece_brier(pred_onehot.cpu(), seg.cpu())
    eces.append(ece)
    briers.append(brier)

    # 4. 关联分析
    entropy = entropy_from_prob(pred.cpu())
    uncertainty_list.append(entropy.numpy().flatten())
    entropy=entropy.cpu().numpy().flatten()
    err_mask = (pred_labels.squeeze(1) != seg.squeeze(1)).cpu().numpy().flatten()
    error_list.append(err_mask)

    # 5. 聚合
    dice = dice_metric.aggregate().cpu().numpy()   # [C]
    hd   = hd_metric.aggregate().cpu().numpy()
    iou  = dice / (2 - dice + 1e-8)
    mean_nll   = np.mean(nlls)
    mean_ece   = np.mean(eces)
    mean_brier = np.mean(briers)
    # overlap_k, recall_k = overlap_recall_k(uncertainty_list, error_list, k=k)
    overlap_k, recall_k = overlap_recall_k(entropy, err_mask, k=k)


    return {
        "dice": dice, "iou": iou, "hd95": hd,
        "nll": mean_nll, "ece": mean_ece, "brier": mean_brier,
        "overlap@100": overlap_k, "recall@100": recall_k
    }

if __name__=='__main__':
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    temp_pred=torch.randn(size=(1,4,128,128,128))
    temp_seg=torch.randint(0,3,size=(1,1,128,128,128))
    result=compute_all_metrics(temp_pred,temp_seg,device)
    print('overlap:',result['overlap@100'])
    print('recall:',result['recall@100'])

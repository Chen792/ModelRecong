import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os

from IPython.core.pylabtools import figsize
from sklearn.calibration import calibration_curve
import seaborn as sns
styles = [
    {'color': 'blue', 'label': 'background 0'},
    {'color': 'red', 'label': 'Necrosis 1'},
    {'color': 'green','label': 'Edema 2'},
    {'color': 'orange','label': 'enhanced 3'},
]
# 生成需要的所有图
"""
成对热力图（Baseline vs 方法）
差值热力图（ΔU=U_base−U_method）
高不确定区域面积对比（Top-k%）
错误重合可视化（不确定性是否聚焦错误）
不确定性分布对比（小提琴/直方图)
Reliability、Risk-Coverage、瀑布图
:return:
"""
def plot_risk_coverage(probs, labels, ax, label):
    """
    probs: numpy array, shape [N, C] 预测概率 随机采样 N=10000 0.1.2.3
    labels: numpy array, shape [N,] 真实类别 随机采样 N=10000 0.1.2.3
    """
    confidences = np.max(probs, axis=1)
    correct = (np.argmax(probs, axis=1) == labels)
    sorted_idx = np.argsort(-confidences)
    risks, coverages = [], []
    for i in range(1, len(sorted_idx) + 1):
        coverages.append(i / len(sorted_idx))
        risks.append(1 - correct[sorted_idx[:i]].mean())
    ax.plot(coverages, risks, label=label)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk (1-Acc)")
    ax.set_title("Risk–Coverage Curve")
    ax.legend()

def plot_reliability(probs, labels, ax, label,line_style , n_bins=15):
    """
    probs: numpy array, shape [N, C] 预测概率 随机采样 N=10000 0.1.2.3
    labels: numpy array, shape [N,] 真实类别 随机采样 N=10000 0.1.2.3
    """
    #对每个类别分别计算reliability
    for i in range(probs.shape[1]):
        style=styles[i]
        y_true_i = (i== labels).astype(int)
        prob_i=probs[:,i]
        frac_pos, mean_pred = calibration_curve(y_true_i, prob_i, n_bins=n_bins)
        ax.plot(mean_pred, frac_pos, line_style, label=style['label'],color=style['color'])
    ax.plot([0, 1], [0, 1], "k--", label='Perfect Calibration')
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{label} Reliability Diagram")
    ax.legend()

def plot_top_k_uncertainty(flatten_base,flatten_method,method_name1,method_name2,ax,k=0.05):
    N_base = flatten_base.size
    N_method = flatten_method.size
    topk_num_base = int(k * N_base)
    topk_num_method = int(k * N_method)

    # 严格 top-k 区域
    topk_base_values = np.sort(flatten_base)[-topk_num_base:]
    topk_method_values = np.sort(flatten_method)[-topk_num_method:]

    # 面积比例
    topk_base_ratio = topk_base_values.size / N_base
    topk_method_ratio = topk_method_values.size / N_method

    # top-k 平均不确定性
    topk_base_mean = topk_base_values.mean()
    topk_method_mean = topk_method_values.mean()

    # 绘制 Bar 图
    ax.bar(
        [method_name1 , method_name2],
        [topk_base_mean, topk_method_mean],
        color=["#ff7f0e", "#1f77b4"],
        alpha=0.7
    )
    ax.set_title(f"Top-{int(k * 100)}% Uncertainty Area Ratio")
    ax.set_ylabel("Ratio")
    ax.set_ylim(0, max(topk_base_mean, topk_method_mean) * 1.2)


def plot_all_map(image,U_base, U_method, err_mask_base, err_mask_method,probs_base, probs_method, labels,methodName1,methodName2, save_dir=None):
    """

    :param U_base: baseline的uncertainty H,W,D
    :param U_method: 需要进行对比的model的uncertainty H,W,D
    :param err_mask_base: baseline的错误掩码 H,W,D
    :param err_mask_method: 需要对比的model的错误掩码 H,W,D
    :param probs_base: baseline的预测概率图 C,H,W,D
    :param probs_method: 需要对比的model的预测概率图 C,H,W,D
    :param labels: GT标签 H,W,D
    :param save_dir: 如果想保存图片的图片路径 str
    :return: None
    """
    delta_U=U_base-U_method
    mid_slice=delta_U.shape[-1]//2
    flatten_base=U_base.flatten()
    flatten_method=U_method.flatten()


    fig,axes=plt.subplots(4,3,figsize=(16,12))
    ax=axes.ravel()

    #热力图绘制
    #baseline热力图
    sns.heatmap(U_base[...,mid_slice],cmap='magma',ax=ax[0],cbar=False)
    ax[0].set_title(f'{methodName1} heatmap')

    #对比的热力图
    sns.heatmap(U_method[...,mid_slice],cmap='magma',ax=ax[1],cbar=False)
    ax[1].set_title(f'{methodName2} heatmap')

    #差值热力图
    sns.heatmap(delta_U[..., mid_slice], cmap="coolwarm", center=0, ax=ax[2],cbar=True)
    ax[2].set_title("diff heatmap")

    #top-k不确定性
    # k = 0.05
    # th_base = np.percentile(flatten_base, 100 * (1 - k))
    # th_method = np.percentile(flatten_method, 100 * (1 - k))
    # topk_base = (flatten_base >= th_base).mean()
    # topk_method = (flatten_method >= th_method).mean()
    # ax[3].bar(["Baseline", f"{methodName}"], [topk_base, topk_method], color=["#ff7f0e", "#1f77b4"])
    # ax[3].set_title(f"Top-{int(k * 100)}% Uncertainty Area Ratio")
    plot_top_k_uncertainty(flatten_base,flatten_method,methodName1,methodName2,ax[3],0.05)

    #错误重合可视化
    overlap_base = (U_base * err_mask_base).sum() / U_base.sum()
    overlap_method = (U_method * err_mask_method).sum() / U_method.sum()
    ax[4].bar([f"{methodName1}", f"{methodName2}"], [overlap_base, overlap_method], color=["#ff7f0e", "#1f77b4"])
    ax[4].set_title("Uncertainty–Error Overlap")

    #不确定性分布对比
    N=min(len(flatten_base),10000)
    np.random.seed(42)
    idx=np.random.choice(len(flatten_base),N,replace=False)
    sample_base_flatten=flatten_base[idx]
    sample_method_flatten=flatten_method[idx]
    sample_labels = labels.cpu().numpy().flatten()[idx]

    sns.violinplot(data=[sample_base_flatten, sample_method_flatten],
                   ax=ax[5], palette=["#ff7f0e", "#1f77b4"])
    ax[5].set_xticklabels([f"{methodName1}", f"{methodName2}"])
    ax[5].set_title("Uncertainty Distribution")

    probs_flatten_base=probs_base.reshape(probs_base.shape[0],-1).T[idx]
    probs_flatten_method=probs_method.reshape(probs_method.shape[0],-1).T[idx]

    #Reliability
    plot_reliability(probs_flatten_base,sample_labels,ax[6],label=f'{methodName1}',line_style='-')   #baseline
    plot_reliability(probs_flatten_method,sample_labels,ax[6],label=f'{methodName2}',line_style='*')  #method

    #risk_coverage
    plot_risk_coverage(probs_flatten_base, sample_labels, ax[7], label=f"{methodName1}")
    plot_risk_coverage(probs_flatten_method, sample_labels, ax[7], label=f"{methodName2}")

    #waterfall
    mean_u_base = np.mean(U_base)
    mean_u_method = np.mean(U_method)
    ax[8].bar([f"{methodName1}", f"{methodName2}"], [mean_u_base, mean_u_method], color=["#ff7f0e", "#1f77b4"])
    ax[8].set_title("Mean Uncertainty (Waterfall)")

    #original img
    ax[9].imshow(image[0,0,:,:,mid_slice].cpu(),cmap='gray')
    ax[9].set_title('Original Image')
    ax[9].axis('off')

    #segment img
    ax[10].imshow(labels[:, :, mid_slice].cpu(), cmap='tab20')
    ax[10].set_title('Ground Truth')
    ax[10].axis('off')


    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir, dpi=300)
    plt.show()




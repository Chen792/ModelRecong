import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from sklearn.calibration import calibration_curve
import seaborn as sns
import pandas as pd
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

def safe_clear(ax):
    ax.cla()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
def plot_risk_coverage(probs1,probs2,labels, ax, label1,label2):
    """
    probs: numpy array, shape [N, C] 预测概率 随机采样 N=10000 0.1.2.3
    labels: numpy array, shape [N,] 真实类别 随机采样 N=10000 0.1.2.3
    """
    safe_clear(ax)
    confidences1 = np.max(probs1, axis=1)
    predictions1 = np.argmax(probs1, axis=1)
    correct1 = (predictions1 == labels)

    confidences2 = np.max(probs2, axis=1)
    predictions2 = np.argmax(probs2, axis=1)
    correct2 = (predictions2 == labels)

    # 按置信度降序排列
    sorted_indices1 = np.argsort(confidences1)[::-1]
    sorted_correct1 = correct1[sorted_indices1]

    sorted_indices2 = np.argsort(confidences2)[::-1]
    sorted_correct2 = correct2[sorted_indices2]

    coverages = np.linspace(0, 1, 100)
    risks1 = []
    risks2 = []

    for coverage in coverages:
        n_include1 = int(coverage * len(sorted_correct1))
        n_include2 = int(coverage * len(sorted_correct2))
        if n_include1 > 0:
            risk1 = 1 - sorted_correct1[:n_include1].mean()
        else:
            risk1 = 1.0
        if n_include2 > 0:
            risk2 = 1 - sorted_correct2[:n_include2].mean()
        else:
            risk2 = 1.0

        risks1.append(risk1)
        risks2.append(risk2)

    ax.plot(coverages, risks1, label=label1, linewidth=2)
    ax.plot(coverages, risks2, label=label2, linewidth=2)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk (1 - Accuracy)",labelpad=30)
    ax.set_title("Risk-Coverage Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_reliability(probs1,probs2,labels, ax, label1,label2 , n_bins=15):
    """
    probs: numpy array, shape [N, C] 预测概率 随机采样 N=10000 0.1.2.3
    labels: numpy array, shape [N,] 真实类别 随机采样 N=10000 0.1.2.3
    """
    #对每个类别分别计算reliability
    safe_clear(ax)
    for i in range(probs1.shape[1]):
        style=styles[i]
        y_true_i = (i== labels).astype(int)
        prob_i=probs1[:,i]
        frac_pos, mean_pred = calibration_curve(y_true_i, prob_i, n_bins=n_bins)
        ax.plot(mean_pred, frac_pos, '-', label=style['label'],color=style['color'])
        y_true_ii = (i == labels).astype(int)
        prob_ii = probs2[:, i]
        frac_pos2, mean_pred2 = calibration_curve(y_true_ii, prob_ii, n_bins=n_bins)
        ax.plot(mean_pred2, frac_pos2, '*', label=style['label'], color=style['color'])
    ax.plot([0, 1], [0, 1], "k--", label='Perfect Calibration')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy",labelpad=30)
    ax.set_title(f"{label1},{label2} Reliability Diagram")
    ax.legend()

def plot_top_k_uncertainty(flatten_base,flatten_method,method_name1,method_name2,ax,k=0.05):
    safe_clear(ax)
    thresh_base = np.percentile(flatten_base, 100 * (1 - k))
    thresh_method = np.percentile(flatten_method, 100 * (1 - k))
    # 计算平均不确定性
    mean_base = flatten_base[flatten_base >= thresh_base].mean()
    mean_method = flatten_method[flatten_method >= thresh_method].mean()

    # 绘制 Bar 图
    ax.bar(
        [0 , 1],
        [mean_base, mean_method],
        color=["#ff7f0e", "#1f77b4"],
        alpha=0.7
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels([method_name1, method_name2])
    ax.set_title(f"Top-{int(k * 100)}% Uncertainty")
    ax.set_ylabel("Mean uncertainty",labelpad=30)
    ax.grid(True,alpha=0.3)
    # ax.set_ylim(0, max(topk_base_mean, topk_method_mean) * 1.2)


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


    fig,axes=plt.subplots(4,4,figsize=(20,16))
    gs=GridSpec(4,4,figure=fig)
    # 第一行：热力图对比
    ax1 = fig.add_subplot(gs[0, 0])  # Baseline热力图
    ax2 = fig.add_subplot(gs[0, 1])  # Method热力图
    ax3 = fig.add_subplot(gs[0, 2])  # 差值热力图
    ax4 = fig.add_subplot(gs[0, 3])  # 原始图像

    # 第二行：统计指标
    ax5 = fig.add_subplot(gs[1, 0])  # Top-k分析
    ax6 = fig.add_subplot(gs[1, 1])  # 错误重合
    ax7 = fig.add_subplot(gs[1, 2])  # 不确定性分布
    ax8 = fig.add_subplot(gs[1, 3])  # 平均不确定性

    # 第三行：校准曲线
    ax9 = fig.add_subplot(gs[2, 0])  # Reliability Diagram
    ax10 = fig.add_subplot(gs[2, 1])  # Risk-Coverage

    # 第四行：分割结果
    ax11 = fig.add_subplot(gs[3, 0])  # 真实结果
    ax12 = fig.add_subplot(gs[3, 1])  # method1预测分割结果
    ax13 = fig.add_subplot(gs[3, 2])  # method2预测分割结果
    ax14 = fig.add_subplot(gs[3, 3])  # 分割区域对比
    for ax in [ax1, ax2, ax3, ax4,ax5, ax6, ax7, ax8, ax9, ax10,ax11,ax12,ax13,ax14]:
        safe_clear(ax)
    #热力图绘制

    #统一颜色范围
    vmin = min(U_base.min(), U_method.min())
    vmax = max(U_base.max(), U_method.max())

    #baseline热力图
    sns.heatmap(U_base[...,mid_slice],cmap='magma',ax=ax1,cbar=False,vmin=vmin,vmax=vmax)
    ax1.set_title(f'{methodName1} heatmap')
    ax1.axis('off')

    #对比的method热力图
    sns.heatmap(U_method[...,mid_slice],cmap='magma',ax=ax2,cbar=False,vmin=vmin,vmax=vmax)
    ax2.set_title(f'{methodName2} heatmap')
    ax2.axis('off')

    #差值热力图
    sns.heatmap(delta_U[..., mid_slice], cmap="coolwarm", center=0, ax=ax3,cbar=True,vmin=vmin,vmax=vmax)
    ax3.set_title("diff heatmap")
    ax3.axis('off')

    #原图
    ax4.imshow(image[0, 0, :, :, mid_slice].cpu(), cmap='gray')
    ax4.set_title('Original Image')
    ax4.axis('off')

    #top-k不确定性
    # k = 0.05
    # th_base = np.percentile(flatten_base, 100 * (1 - k))
    # th_method = np.percentile(flatten_method, 100 * (1 - k))
    # topk_base = (flatten_base >= th_base).mean()
    # topk_method = (flatten_method >= th_method).mean()
    # ax[3].bar(["Baseline", f"{methodName}"], [topk_base, topk_method], color=["#ff7f0e", "#1f77b4"])
    # ax[3].set_title(f"Top-{int(k * 100)}% Uncertainty Area Ratio")
    plot_top_k_uncertainty(flatten_base,flatten_method,methodName1,methodName2,ax5,0.05)


    #错误重合可视化
    overlap_base = (U_base * err_mask_base).sum() / U_base.sum()
    overlap_method = (U_method * err_mask_method).sum() / U_method.sum()
    ax6.bar([0,1], [overlap_base, overlap_method], color=["#ff7f0e", "#1f77b4"])
    ax6.set_xticks([0, 1])
    ax6.set_xticklabels([f'{methodName1}', f'{methodName2}'])
    ax6.set_title("Uncertainty–Error Overlap")
    ax6.set_ylabel("Overlap Ratio",labelpad=30)
    ax6.grid(True, alpha=0.3)

    #不确定性分布对比
    N = min(len(flatten_base), 10000)
    np.random.seed(42)
    idx = np.random.choice(len(flatten_base), N, replace=False)
    sample_base_flatten = flatten_base[idx]
    sample_method_flatten = flatten_method[idx]
    sample_labels = labels.cpu().numpy().flatten()[idx]

    # 绘制小提琴图
    uncertainty_data = pd.DataFrame({
        'uncertainty': np.concatenate([sample_base_flatten, sample_method_flatten]),
        'method': np.concatenate([
            np.full(len(sample_base_flatten), methodName1),
            np.full(len(sample_method_flatten), methodName2)
        ])
    })

    # 绘制小提琴图，调整参数确保完整显示
    sns.violinplot(data=uncertainty_data, x='method', y='uncertainty',
                   ax=ax7, palette=["#ff7f0e", "#1f77b4"],
                   cut=0,  # 确保小提琴图完整显示，不截断
                   hue='method',
                   inner="quartile",  # 显示内部四分位数
                   density_norm='width')  # 根据数据量调整宽度
    # sns.violinplot(data=[sample_base_flatten, sample_method_flatten],
    #                ax=ax7, palette=["#ff7f0e", "#1f77b4"])

    # 设置 x 轴标签
    ax7.set_xticks([0, 1])  # 明确设置刻度位置
    ax7.set_xticklabels([methodName1, methodName2], fontsize=10)  # 设置方法名称标签

    ax7.set_title("Uncertainty Distribution")
    ax7.set_ylabel("Uncertainty", fontsize=10)
    ax7.set_xlabel("")
    x_min, x_max = ax7.get_xlim()
    ax7.set_xlim(x_min - 0.5, x_max + 0.5)

    probs_flatten_base=probs_base.reshape(probs_base.shape[0],-1).T[idx]
    probs_flatten_method=probs_method.reshape(probs_method.shape[0],-1).T[idx]

    #waterfall
    mean_u_base = np.mean(U_base)
    mean_u_method = np.mean(U_method)
    ax8.bar([0,1], [mean_u_base, mean_u_method], color=["#ff7f0e", "#1f77b4"])
    ax8.set_xticks([0, 1])
    ax8.set_xticklabels([f"{methodName1}", f"{methodName2}"])
    ax8.set_title("Mean Uncertainty (Waterfall)")

    # Reliability
    plot_reliability(probs_flatten_base,probs_flatten_method, sample_labels,ax9, label1=f'{methodName1}',label2=f'{methodName2}' )  # baseline

    #risk_coverage
    plot_risk_coverage(probs_flatten_base.detach().cpu().numpy(),probs_flatten_method.detach().cpu().numpy(), sample_labels, ax10, label1=f"{methodName1}",label2=f"{methodName2}")

    #segment img
    pred_base = torch.argmax(probs_base, dim=0)[:, :, mid_slice].cpu()
    pred_method = torch.argmax(probs_method, dim=0)[:, :, mid_slice].cpu()
    gt_slice = labels[:, :, mid_slice].cpu()

    ax11.imshow(gt_slice, cmap='tab20')
    ax11.set_title('Ground Truth')
    ax11.axis('off')

    ax12.imshow(pred_base, cmap='tab20')
    ax12.set_title('method1 segment')
    ax12.axis('off')

    ax13.imshow(pred_method, cmap='tab20')
    ax13.set_title('method2 segment')
    ax13.axis('off')
    err_base = (pred_base != gt_slice).float()
    err_method = (pred_method != gt_slice).float()

    # 创建复合图像：红色表示Baseline错误，绿色表示Method错误，黄色表示两者都错
    error_comparison = np.zeros((*err_base.shape, 3))
    error_comparison[..., 0] = err_base.numpy()  # 红色通道 - Baseline错误
    error_comparison[..., 1] = err_method.numpy()  # 绿色通道 - Method错误
    ax14.imshow(error_comparison)
    ax14.set_title('Error Region Comparison\n(Red: Baseline, Green: Method, Yellow: Both)')
    ax14.axis('off')
    for ax in fig.axes:
        # 旋转坐标刻度，防止文本重叠
        ax.tick_params(axis='x', labelrotation=45)
        ax.tick_params(axis='y', labelrotation=0)

        # 缩小字体以减少重叠
        ax.tick_params(axis='both', labelsize=8)


    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir, dpi=300)
    # plt.show()




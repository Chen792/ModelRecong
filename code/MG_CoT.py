import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from predict_func import Single_Model_pred,Mutil_Model_pred

def MG_CoT_pred(image,model, device, mode='baseline', patch_size=(32, 32, 32), smoothing_sigma=1.0):
    """
    MG-CoT 推理
    :param image: [B,C,H,W,D]
    :param device: torch.device
    :param model_path: 模型路径
    :param mode: 'single' 或 'ensemble'
    :param patch_size: 高不确定patch大小
    :param smoothing_sigma: 平滑系数
    :param T: MC-dropout采样次数（单模型时可用）
    :return: refined_probs, refined_seg, entropy_map
    """
    image = image.to(device)

    # --- 1. 得到基础预测 ---
    if mode == 'baseline':
        probs, _ = Single_Model_pred(image, device, model, False)
    elif mode == 'al':
        probs, _ = Single_Model_pred(image, device, model, False)
    elif mode == 'ensemble':
        probs, _ = Mutil_Model_pred(image, device, model, False)
    else:
        probs, _ = Mutil_Model_pred(image, device, model, False)

    B, C, H, W, D = probs.shape

    # --- 2. 计算像素级熵 ---
    entropy_map = -(probs * torch.log(probs + 1e-8)).sum(1)  # [B,H,W,D]

    # --- 3. Patch级修正 ---
    ph, pw, pd = patch_size
    refined_probs = probs.clone()

    for b in range(B):
        for x in range(0, H, ph):
            for y in range(0, W, pw):
                for z in range(0, D, pd):
                    patch_entropy = entropy_map[b,
                                    x:min(x + ph, H),
                                    y:min(y + pw, W),
                                    z:min(z + pd, D)]
                    # 如果是高不确定性patch，做平滑
                    if patch_entropy.mean() > np.percentile(entropy_map[b].cpu().numpy(), 75):
                        for c in range(C):
                            refined_probs[b, c,
                            x:min(x + ph, H),
                            y:min(y + pw, W),
                            z:min(z + pd, D)
                            ] = torch.tensor(
                                gaussian_filter(
                                    refined_probs[b, c,
                                    x:min(x + ph, H),
                                    y:min(y + pw, W),
                                    z:min(z + pd, D)
                                    ].cpu().numpy(),
                                    sigma=smoothing_sigma
                                )
                            ).to(device)

    refined_seg = refined_probs.argmax(1)  # [B,H,W,D]

    return refined_probs, refined_seg, entropy_map

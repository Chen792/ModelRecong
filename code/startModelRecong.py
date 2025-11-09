import os
import random

from monai.data import DataLoader
from BraTSDataset.getData import getDatasetAndLoaderAndOthers
import torch
from predict_func import Single_Model_pred,Mutil_Model_pred,calculate_dice_score
from cal_uncertainty_probs import cal_uncertainty_online,cal_prob
from copy import deepcopy
from monai.networks.nets import UNet
from testModelAndShowImg import plot_all_map


def get_normalized_weights(scores):
    """根据性能分数计算归一化权重。"""
    total_score = sum(scores)
    num_models = len(scores)

    if total_score == 0:
        # 防止除以零，如果所有模型都很差，退化为简单平均
        weights = [1.0 / num_models] * num_models
        print("Warning: All model scores are zero. Using simple averaging.")
    else:
        # 权重 = (该模型的Dice Score) / (所有模型的Dice Score之和)
        weights = [score / total_score for score in scores]

    return weights
if __name__=='__main__':

    #加载模型，进行推理预测
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset, val_dataset, test_dataset, transforms, cases=getDatasetAndLoaderAndOthers()
    #只进行推理
    test_loader=DataLoader(test_dataset,batch_size=1,shuffle=False)
    val_loader=DataLoader(val_dataset,batch_size=1,shuffle=False)

    #随机3个需要展示图片的plot
    sample_indices = random.sample(range(len(test_loader.dataset)),3)

    model = UNet(spatial_dims=3,
                 in_channels=4,
                 out_channels=4,
                 channels=(32, 64, 128, 256, 512),
                 strides=(2, 2, 2, 2),
                 num_res_units=2,
                 dropout=0.2)
    baseline_model = deepcopy(model).to(device)
    baseline_model.load_state_dict(torch.load('../save_model/baseline/model.pth'))

    al_model = deepcopy(model).to(device)
    al_model.load_state_dict(torch.load('../save_model/al_model/model.pth'))
    #
    ensemble_models = []
    dropout_rates = 0.2

    for i in range(5):
        ensemble_model = UNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=4,
            channels=[32,64,128,256,512],
            strides=(2, 2, 2, 2),
            num_res_units=2,
            dropout=dropout_rates
        ).to(device)
        ensemble_model.load_state_dict(torch.load(f'../save_model/ensemble/model{i}.pth'))
        ensemble_models.append(ensemble_model)
    #
    al_ensemble_models = []
    for i in range(5):
        al_ensemble_model = UNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=4,
            channels=[32,64,128,256,512],
            strides=(2, 2, 2, 2),
            num_res_units=2,
            dropout=dropout_rates
        ).to(device)
        al_ensemble_model.load_state_dict(torch.load(f'../save_model/al_and_ensemble/model{i}.pth'))
        al_ensemble_models.append(al_ensemble_model)

    os.makedirs(f'../SaveImg',exist_ok=True)
    #保存所有推理过程的中间结果，最后随机输出三张图
    ens_dice_scores = [calculate_dice_score(model, val_loader, device) for model in ensemble_models]
    ens_weights = get_normalized_weights(ens_dice_scores)

    al_ens_dice_scores = [calculate_dice_score(model, val_loader, device) for model in al_ensemble_models]
    al_ens_weights = get_normalized_weights(al_ens_dice_scores)

    for idx,batch in enumerate(test_loader):
        image=batch['image'].to(device) #[B,C,H,W,D]
        seg=batch['seg'].to(device) #[B,C,H,W,D]
        seg=seg.squeeze()
        case_idx=batch['case_idx']
        # print('img shape is',image.shape)
        # print('seg shape is',seg.shape)
        # baseline的推理

        baseline_pred,baseline_prob_labels=Single_Model_pred(image,device,baseline_model,False)
        baseline_prob_labels=baseline_prob_labels.squeeze(0)

        #baseline的错误掩码

        baseline_error_mask=(baseline_prob_labels!=seg)

        baseline_list=[]

        #计算概率

        baseline_probs=cal_prob(baseline_pred)

        #计算不确定性 ——3次取平均

        baseline_uncertainty=cal_uncertainty_online(baseline_model,image,device,3,False)

        # al_model的推理

        al_model_pred, al_model_prob_labels = Single_Model_pred(image, device, al_model,False)
        al_model_list = []
        al_model_prob_labels=al_model_prob_labels.squeeze(0)
        #al_model的错误掩码

        al_model_error_mask = (al_model_prob_labels != seg)

        # 计算概率

        al_model_probs = cal_prob(al_model_pred)

        # 计算不确定性 ——3次取平均

        al_model_uncertainty=cal_uncertainty_online(al_model,image,device,3,False)

        # al_model_uncertainty = cal_uncertainty(preds)

        #ens_model的推理

        ens_model_pred, ens_model_prob_labels = Mutil_Model_pred(image, device, ensemble_models,False,ens_weights)
        ens_model_list = []
        ens_model_prob_labels=ens_model_prob_labels.squeeze(0)

        # ens_model的错误掩码

        ens_model_error_mask = (ens_model_prob_labels != seg)

        # 计算概率

        ens_model_probs = cal_prob(ens_model_pred)

        # 计算不确定性 ——3次取平均

        ens_model_uncertainty = cal_uncertainty_online(ensemble_models,image,device,3,True)

        # al_ens_model的推理

        al_ens_model_pred, al_ens_model_prob_labels = Mutil_Model_pred(image, device, al_ensemble_models,False,al_ens_weights)
        al_ens_model_list = []
        al_ens_model_prob_labels=al_ens_model_prob_labels.squeeze(0)

        # al_ens_model的错误掩码

        al_ens_model_error_mask = (al_ens_model_prob_labels != seg)

        # 计算概率

        al_ens_model_probs = cal_prob(al_ens_model_pred)

        # 计算不确定性 ——3次取平均

        al_ens_model_uncertainty=cal_uncertainty_online(al_ensemble_models,image,device,3,True)

        if idx not in sample_indices:
            torch.cuda.empty_cache()
            continue
        os.makedirs(f'../SaveImg',exist_ok=True)
        #baseline vs almodel
        plot_all_map(image,baseline_uncertainty,al_model_uncertainty,baseline_error_mask,al_model_error_mask,baseline_probs,al_model_probs,seg,'baseline','almodel',f'../SaveImg/baseline_vs_al_model_{idx}')

        #baseline vs ensemble
        plot_all_map(image,baseline_uncertainty,ens_model_uncertainty,baseline_error_mask,ens_model_error_mask,baseline_probs,ens_model_probs,seg,'baseline','ensemble',f'../SaveImg/baseline_vs_ensemble_{idx}')

        #baseline vs al_ensemble
        plot_all_map(image,baseline_uncertainty,al_ens_model_uncertainty,baseline_error_mask,al_ens_model_error_mask,baseline_probs,al_ens_model_probs,seg,'baseline','al_ensemble',f'../SaveImg/baseline_vs_al_ensemble_{idx}')











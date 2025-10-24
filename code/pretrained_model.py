import copy

import monai
from monai.data import DataLoader,Dataset
from torch.utils.data import random_split
import torch
from monai.networks.nets import UNet
from train_step import train_epochs
from alcode import active_learning_loop
from ensemble_train import ensemble_train_model
import os,glob
from monai.transforms import (
    LoadImaged,Compose,ToTensord,EnsureChannelFirstd
)
import pickle
if __name__=='__main__':
    preprocessed_cases = []
    output_dir=r'../data/BraTS2021_preprocess'
    for i,image_file in enumerate(glob.glob(os.path.join(output_dir, "*_t1_pre.nii.gz"))):
        case_id = os.path.basename(image_file).replace("_t1_pre.nii.gz", "")
        seg_file = os.path.join(output_dir, f"{case_id}_seg_pre.nii.gz")
        preprocessed_cases.append({
            'image': image_file,
            'seg': seg_file,
            'case_idx':i
        })

    transforms = Compose([
        LoadImaged(keys=['image', 'seg']),
        EnsureChannelFirstd(keys=['image', 'seg']),
        ToTensord(keys=['image', 'seg'])
    ])
    BraTSDataset = Dataset(preprocessed_cases,transforms)
    dataset_size = len(BraTSDataset)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(BraTSDataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=UNet(spatial_dims=3,
               in_channels=4,
               out_channels=4,
               channels=(32,64,128,256,512),
               strides=(2,2,2,2),
               num_res_units=2,
               dropout=0.2).to(device)

    loss=monai.losses.DiceLoss(to_onehot_y=True,softmax=True)
    optim=torch.optim.AdamW(model.parameters(),lr=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=20,eta_min=1e-6)


    #保存模型，方便后续启动

    #baseline
    baseline=copy.deepcopy(model)
    baseline=train_epochs(baseline,train_loader,val_dataset,device,10,None)
    torch.save(baseline.state_dict(),'../save_model/baseline/model.pth')
    #al
    al_model=copy.deepcopy(model)
    al_model,labeled_idx=active_learning_loop(al_model,train_dataset,val_dataset,device,transforms)
    with open(f'../high_value_dataset/labeled_idx.pkl','wb') as f:
        pickle.dump(labeled_idx,f)
    torch.save(al_model.state_dict(),'../save_model/al_model/model.pth')
    #ensemble
    ens_model=copy.deepcopy(model)
    ens_model=ensemble_train_model(train_dataset)
    for i,m in enumerate(ens_model):
        torch.save(m.state_dict(), f'../save_model/ensemble/model{i}.pth')

    # al+ensemble
    al_ens_model=copy.deepcopy(model)
    with open(f'../high_value_dataset/labeled_idx.pkl','rb') as f:
        labeled_idx=pickle.load(f)
    train_set = torch.utils.data.Subset(BraTSDataset, list(labeled_idx))
    al_ens_model=ensemble_train_model(train_set)
    for i,m in enumerate(al_ens_model):
        torch.save(m.state_dict(), f'../save_model/al_and_ensemble/model{i}.pth')
from monai.networks.nets import UNet
from BraTSDataset.getData import getDatasetAndLoaderAndOthers
import torch
from monai.data import DataLoader
import os
from train_step import train_epochs
from torch.nn import init

device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def weight_reset(m):
    if isinstance(m, (torch.nn.Conv3d, torch.nn.Linear)):
        init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)
def ensemble_train_model(dataset):
    models=[]
    print('start ensemble train')
    train_dataset,val_dataset,test_dataset,transforms,cases=getDatasetAndLoaderAndOthers()
    train_loader=DataLoader(dataset,batch_size=1,shuffle=True)
    #集成5个模型
    for i in range(5):
        print(f'{i} model start')
        torch.manual_seed(42 + i*10)
        model = UNet(spatial_dims=3,
               in_channels=4,
               out_channels=4,
               channels=(32 ,64 ,128 ,256 ,512),
               strides=(2 ,2 ,2 ,2),
               num_res_units=2,
               dropout=0.2).to(device)
        model.apply(weight_reset)
        es_model=train_epochs(model,train_loader,val_dataset,device,20,None)
        models.append(es_model)
        save_path = f"../save_model/ensemble/model{i}.pth"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(es_model.state_dict(), save_path)
        print(f"模型参数已保存到 {save_path}")
    return models
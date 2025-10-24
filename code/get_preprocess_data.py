import os,glob
import numpy as np
from monai.transforms import (
    Lambdad,LoadImaged,Spacingd,NormalizeIntensityd,CropForegroundd,SaveImaged,Compose,SpatialPadd,CenterSpatialCropd,EnsureChannelFirstd,ConcatItemsd,ToTensord
)
from monai.data import Dataset

#仅运行一次，保存处理后的文件夹，方便后续调用。
if __name__=='__main__':
    #数据路径
    data_dir=r'E:\DataSet\ModelRecong\BraTS\TrainData'
    #预处理后的数据存放路径
    output_dir=r'../data/BraTS2021_preprocess'
    #组合路径
    os.makedirs(output_dir,exist_ok=True)

    #按照文件构建字典列表
    cases=[]
    for p in sorted(glob.glob(os.path.join(data_dir,'*/*_t1.nii.gz'))):
        #找到文件
        #p likes E:/DataSet/ModelRecong/BraTS/TrainData/BraTS2021_00000/BraTS2021_00000_t1.nii.gz
        case_id=p.split('/')[-1].split('_t1')[0]  #case_id likes BraTS2021_00000
        #构造剩下的图像字典
        cases.append({
            't1':os.path.join(data_dir,case_id,f'{case_id}_t1.nii.gz'), #t1图
            't1ce':os.path.join(data_dir,case_id,f'{case_id}_t1ce.nii.gz'), #t1+对比
            't2':os.path.join(data_dir,case_id,f'{case_id}_t2.nii.gz'), #t2图
            'flair':os.path.join(data_dir,case_id,f'{case_id}_flair.nii.gz'), #flair模态
            'seg':os.path.join(data_dir,case_id,f'{case_id}_seg.nii.gz') #seg后图 -label
        })

    #预处理case
    transforms = Compose([
        LoadImaged(keys=['t1', 't1ce', 't2', 'flair', 'seg']),
        EnsureChannelFirstd(keys=['t1', 't1ce', 't2', 'flair', 'seg']),  # 确保 CHW(D) 格式
        Spacingd(keys=['t1', 't1ce', 't2', 'flair', 'seg'],
                 pixdim=(1.0, 1.0, 1.0),
                 mode=("bilinear", "bilinear", "bilinear", "bilinear", "nearest")),
        NormalizeIntensityd(keys=['t1', 't1ce', 't2', 'flair'], nonzero=True, channel_wise=True),
        CropForegroundd(keys=['t1', 't1ce', 't2', 'flair', 'seg'], source_key='t1'),
        SpatialPadd(keys=['t1', 't1ce', 't2', 'flair', 'seg'], spatial_size=(128, 128, 128), method='symmetric'),
        CenterSpatialCropd(keys=['t1', 't1ce', 't2', 'flair', 'seg'], roi_size=(128, 128, 128)),
        Lambdad(keys=["seg"], func=lambda x: np.where(x == 4, 3, x)),

        # 合并 4 模态为 1 个 4-channel image
        ConcatItemsd(keys=['t1', 't1ce', 't2', 'flair'], name='image'),  # <- 关键
        ToTensord(keys=['image', 'seg']),

        # 只保存合并后的 4-channel image 和 label
        SaveImaged(keys=['image', 'seg'],
                   output_dir=output_dir,
                   output_postfix='pre',
                   resample=False,
                   separate_folder=False)  # 所有文件放同一目录，方便管理
    ])

    #构造dataset和dataloader
    BraTSDataset=Dataset(cases,transforms)
    for i,data in enumerate(BraTSDataset):
        print(f"processed case {i+1}/{len(BraTSDataset)}")

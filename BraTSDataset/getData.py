import torch
from monai.data import DataLoader
import glob,os
from monai.transforms import (
    LoadImaged,Spacingd,NormalizeIntensityd,CropForegroundd,SaveImaged,Compose,SpatialPadd,CenterSpatialCropd,EnsureChannelFirstd,ConcatItemsd,ToTensord
)
from torch.utils.data import random_split
from BraTSDataset.BraTSDataset import BraTSDataset
data_dir = r'../data/BraTS2021_preprocess'
# 构造 image-seg 字典
cases = []
images = sorted(glob.glob(os.path.join(data_dir, "*_t1_pre.nii.gz")))
segs   = sorted(glob.glob(os.path.join(data_dir, "*_seg_pre.nii.gz")))

for idx,(img, seg) in enumerate(zip(images, segs)):
    cases.append({"image": img, "seg": seg,'case_idx': idx})

# 定义 transforms（这时候只需要加载 + 转tensor）
transforms = Compose([
    LoadImaged(keys=["image", "seg"]),
    EnsureChannelFirstd(keys=["image", "seg"]),
    ToTensord(keys=["image", "seg"])
])
def getDatasetAndLoaderAndOthers():
    BraTSdataset =BraTSDataset(cases ,transforms)
    dataset_size = len(BraTSdataset)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(BraTSdataset, [train_size, val_size, test_size])
    generator=torch.Generator().manual_seed(42)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_dataset,val_dataset,test_dataset,transforms,cases
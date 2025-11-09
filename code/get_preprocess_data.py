import os
import glob
import numpy as np
from monai.transforms import (
    Lambdad, LoadImaged, Spacingd, NormalizeIntensityd, CropForegroundd, SaveImaged,
    Compose, SpatialPadd, CenterSpatialCropd, EnsureChannelFirstd, ConcatItemsd,
    ToTensord, EnsureTyped,DeleteItemsd
)
from monai.data import Dataset
import nibabel as nib

# 仅运行一次，保存处理后的文件夹，方便后续调用。
if __name__ == '__main__':
    # 数据路径：使用 os.path.join 确保路径兼容性
    data_dir = r'E:\DataSet\ModelRecong\BraTS\TrainData'
    # 预处理后的数据存放路径
    output_dir = os.path.join('..', 'data', 'BraTS2021_preprocess')
    os.makedirs(output_dir, exist_ok=True)

    # 按照文件构建字典列表
    cases = []
    search_path = os.path.join(data_dir, '*', '*_t1.nii.gz')
    for p in sorted(glob.glob(search_path)):
        # 使用 os.path.basename 和字符串操作来获取 case_id
        base_name = os.path.basename(p)  # e.g., BraTS2021_00000_t1.nii.gz
        case_id = base_name.split('_t1')[0]  # e.g., BraTS2021_00000

        # 构造图像字典
        case_data = {
            't1': os.path.join(data_dir, case_id, f'{case_id}_t1.nii.gz'),
            't1ce': os.path.join(data_dir, case_id, f'{case_id}_t1ce.nii.gz'),
            't2': os.path.join(data_dir, case_id, f'{case_id}_t2.nii.gz'),
            'flair': os.path.join(data_dir, case_id, f'{case_id}_flair.nii.gz'),
            'seg': os.path.join(data_dir, case_id, f'{case_id}_seg.nii.gz'),
            'case_id': case_id  # 保存 case_id
        }
        cases.append(case_data)

    keys = ['t1', 't1ce', 't2', 'flair', 'seg']
    img_keys=['t1', 't1ce', 't2', 'flair']
    # 预处理 transforms
    transforms = Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Spacingd(keys=keys,
                 pixdim=(1.0, 1.0, 1.0),
                 mode=("bilinear",) * 4 + ("nearest",)),
        NormalizeIntensityd(keys=img_keys, nonzero=True, channel_wise=True),
        CropForegroundd(keys=keys, source_key='t1'),
        SpatialPadd(keys=keys, spatial_size=(128, 128, 128), method='symmetric'),
        CenterSpatialCropd(keys=keys, roi_size=(128, 128, 128)),
        Lambdad(keys=["seg"], func=lambda x: np.where(x == 4, 3, x)),

        # 合并 4 模态为 1 个 4-channel image
        ConcatItemsd(keys=img_keys, name='image'),
        DeleteItemsd(keys=img_keys),
        EnsureTyped(keys=['image', 'seg']),
        ToTensord(keys=['image', 'seg']),
    ])

    # 构造dataset
    BraTSDataset = Dataset(cases, transforms)

    # 遍历数据集并保存
    for i, data in enumerate(BraTSDataset):
        case_id = cases[i]['case_id']
        print(f"Processing case {i + 1}/{len(BraTSDataset)}: {case_id}")

        # 获取数据
        image_array = data['image'].numpy()  # 形状: (4, 128, 128, 128)
        seg_array = data['seg'].numpy()  # 形状: (1, 128, 128, 128)

        # 获取affine矩阵
        if 't1_meta_dict' in data:
            affine = data['t1_meta_dict']['affine']
        else:
            affine = np.eye(4)
            print(f"Warning: No affine found for case {case_id}, using identity matrix")

        # 使用nibabel保存
        image_path = os.path.join(output_dir, f"{case_id}_t1_pre.nii.gz")
        seg_path = os.path.join(output_dir, f"{case_id}_seg_pre.nii.gz")

        # 对于4通道图像：需要将通道维度移到最后 (4,128,128,128) -> (128,128,128,4)
        image_array_chn_last = np.moveaxis(image_array, 0, -1)
        seg_array = np.moveaxis(seg_array, 0, -1)

        # 创建NIfTI图像对象并保存
        img_nifti = nib.Nifti1Image(image_array_chn_last, affine)
        seg_nifti = nib.Nifti1Image(seg_array.astype(np.int8), affine)

        nib.save(img_nifti, image_path)
        nib.save(seg_nifti, seg_path)

    print("--- 预处理和保存完成 ---")
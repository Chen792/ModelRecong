from monai.data import Dataset
class BraTSDataset(Dataset):
    def __init__(self, cases, transform):
        super().__init__(cases, transform)
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item["case_idx"] = self.data[idx]["case_idx"]
        return item
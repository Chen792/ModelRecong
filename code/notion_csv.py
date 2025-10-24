import csv
import os

# 保存训练/验证指标到 CSV
def save_metrics_csv(log_file, epoch, train_loss=None, val_metrics=None):
    """
    log_file: CSV路径
    epoch: 当前轮数
    train_loss: 当前训练loss
    val_metrics: dict, 如 {"dice": [..], "iou": [..], "hd95": [..]}
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # 检查文件是否存在
    file_exists = os.path.isfile(log_file)

    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        # 写表头
        if not file_exists:
            header = ["epoch", "train_loss"] + (list(val_metrics.keys()) if val_metrics else [])
            writer.writerow(header)
        # 写内容
        row = [epoch, train_loss] + (list(val_metrics.values()) if val_metrics else [])
        writer.writerow(row)
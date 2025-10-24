import torch
from monai.networks.nets import UNet
import torch.nn.functional as F
import os
#单个模型的推理
def Single_Model_pred(image,device,model,is_MC):
    """

    :param image: 输入的图像
    :param device: 在哪个设备上处理
    :param model_path: 模型的父文件夹路径
    :param mode: 哪种训练？baseline？Active Learning? 0=baseline;1=AL
    :param is_MC: 需不需要进行MC Dropout
    :return: 预测概率
    """
    #MC Dropout层
    # 2. 加载保存的参数
    if is_MC:
        model.train()
    else:
        model.eval()
    with torch.no_grad():
        preds=model(image)
    pred=F.softmax(preds,dim=1) #[B,C,H,W,D]
    prob_labels=torch.argmax(pred,dim=1) #[B,H,W,D]
    del preds
    torch.cuda.empty_cache()
    return pred,prob_labels


#集成模型的推理
def Mutil_Model_pred(image,device,models,is_MC):
    """

    :param image: 输入的图像
    :param device: 在哪个设备上运行
    :param models_path: 集成模型的父文件夹路径
    :param mode: 哪种模式： 0=ensemble; 1=ensemble+al
    :param is_MC: 需不需要train模式来保证MC Dropout
    :return: 预测概率
    """
    pred=None
    for i in range(5):
        model=models[i]
        if is_MC:
            model.train()
        else:
            model.eval()
        with torch.no_grad():
            logits=model(image)
        if pred==None:
            pred=F.softmax(logits,dim=1)/len(models)
        else:
            pred += F.softmax(logits,dim=1)/len(models) #[B,C,H,W,D]
        del logits
        torch.cuda.empty_cache()
    prob_labels = torch.argmax(pred, dim=1) #[B,H,W,D]
    return pred,prob_labels

























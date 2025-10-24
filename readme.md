怎么开始？

由于有些数据过大，因此并没有一起上传到github，有一些文件夹需要自己手动创建。因为os.makedir并没有做的很全面。如果有文件路径不存在等问题，可以看一看对应位置，是不是有的文件夹没有创建。

首先下载项目以后，进行数据集的处理。

下载数据集以后，运行code/get_preprocess_data.py,更改其中的data_dir变量为自己存放数据集的文件路径。以下为示例，具体数据集我会放在群里。

一个BraTS2021的数据由五张照片组成

<img width="767" height="145" alt="image" src="https://github.com/user-attachments/assets/7f9b782b-89a8-43a7-9824-29c373b1c910" />

data_dir改到这个路径就好

<img width="592" height="191" alt="image" src="https://github.com/user-attachments/assets/2df558fe-0f96-4863-bfcc-fb2efd44d9b7" />

运行好后，data文件夹下会出现BraTS2021_preprocess, 每一个包括t1_pre和seg_pre

然后运行code/pretrained_model.py来训练模型。以后如果对应路径下有模型的话可以不用运行。运行后，会创建一个save_model文件夹，下面有4个文件夹分别对应4种模型：baseline、ensemble、active learning、ensemble training with active learning。

最后运行code/startModelRecong.py来测试。运行后会输出并保存一堆图，对应会创建SaveImg文件夹和log文件夹，可以看一下。

文件解析：以下文件夹可能有的在运行结束后并没有，因为有些文件夹是观察中间运行结果的。比如prob_imgs和uncertainty。

BraTSDataset:

对数据集的处理

code:

主要运行代码，后面具体细讲

config:

老师要求项目带有的yaml文件，包括一些实验参数，虽然代码里是定死的，但是还是要有

data:

具体的BraTS2021数据集

已经经过修正，所有数据放在BraTS2021_preprocess文件夹

high_value_dateset:

主动学习学到的subset，方便训练

logs:

保存一些指标 dice iou。。。这部分指标需要进一步修正。

prob_imgs：

预测出来的imgs，方便单独展示

save_model:

最后训练的model

SaveImg：

最后输出的对比图以及要求的图，缺少预测时+CoT的对比图，之后需要加入，以及一些对比图热力图效果不好，极度不好，需要修改模型或者进一步炼丹

uncertainty：

不确定性图，方便单独展示

code具体介绍：

alcode:

经过主动学习的模型

cal_num:

计算相关指标

cal_uncertainty_probs:

计算不确定性和概率图，为了方便展示单一图的效果

ensemble_train:

集成模型

get_preprocess_data:

对初始数据集进行整合

MG-CoT:

初步我设想的CoT

notion_csv:

保存训练中的一些指标

predict_func:

对不同模型进行测试

pretrained_model:

如果一开始没有保存模型，就开始对所有模型进行定义训练(baseline——UNet、集成模型、主动学习、主动学习+集成模型)，还应该加入CoT,还没加，没考虑好CoT用什么方法实现

startModelRecong：

如果已经有了model参数，开始进行测试吧

testModelAndShowImg：

根据预测图和真实图画出对比图热力图等一系列需要展示的图

train_step:

训练的一步

需要解决的问题：

1、测试结果不太好，一些图例效果看起来不好

2、指标有些高，需要进一步炼丹

3、CoT还需要加到预测环节中，而且CoT具体实现细则还没讨论好怎么实现

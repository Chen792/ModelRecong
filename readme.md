运行出现问题了？看最后的几个问题介绍

怎么开始？

因为os.makedir并没有做的很全面。如果有文件路径不存在等问题，可以看一看对应位置，是不是有的文件夹没有创建。

首先下载项目以后，先下载requirement.txt中的库以及对应版本。我所用的python版本是3.10

然后进行数据集的处理。

下载数据集以后，运行code/get_preprocess_data.py,更改其中的data_dir变量为自己存放数据集的文件路径。以下为示例

一个BraTS2021的数据由五张照片组成

<img width="767" height="145" alt="image" src="https://github.com/user-attachments/assets/7f9b782b-89a8-43a7-9824-29c373b1c910" />

data_dir改到这个路径就好

<img width="592" height="191" alt="image" src="https://github.com/user-attachments/assets/2df558fe-0f96-4863-bfcc-fb2efd44d9b7" />

运行好后，data文件夹下会出现BraTS2021_preprocess, 每一个包括t1_pre和seg_pre

然后运行code/pretrained_model.py来训练模型。以后如果对应路径下有模型的话可以不用运行。运行后，会创建一个save_model文件夹，下面有4个文件夹分别对应4种模型：baseline、ensemble、active learning、ensemble training with active learning。

最后运行code/startModelRecong.py来测试。运行后会输出并保存一堆图，对应会创建SaveImg文件夹和log文件夹。

文件解析：以下文件夹可能有的在运行结束后并没有，因为有些文件夹是观察中间运行结果的。比如prob_imgs和uncertainty。

BraTSDataset:

对数据集的处理

code:

主要运行代码，后面具体细讲

config:

包括一些实验参数，虽然代码里是定死的，但是还是要有

data:

具体的BraTS2021数据集

已经经过修正，所有数据放在BraTS2021_preprocess文件夹

high_value_dateset:

主动学习学到的subset，方便训练

logs:

保存一些指标 dice iou。

prob_imgs：

预测出来的imgs，方便单独展示

save_model:

最后训练的model

SaveImg：

最后输出的对比图以及要求的图

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

初步设想的CoT

notion_csv:

保存训练中的一些指标

predict_func:

对不同模型进行测试

pretrained_model:

如果一开始没有保存模型，就开始对所有模型进行定义训练(baseline——UNet、集成模型、主动学习、主动学习+集成模型)

startModelRecong：

如果已经有了model参数，开始进行测试吧

testModelAndShowImg：

根据预测图和真实图画出对比图热力图等一系列需要展示的图

train_step:

训练的一步


问题介绍以及解决办法

1、get_preprocess_data运行以后，data文件夹路径错误：运行代码的时候要保证路径在modelrecong/code下

2、找不到BraTS2021这个包：加入如下代码：
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
本质上是因为sys的path里没有modelrecog这个路径，文件不知道在这个路径下找到对应的包。只知道可以导入code文件夹下的文件，而code同级的文件夹下的就找不到。因此加入这一行代码，这行代码加在一开始运行的文件即可。

3、没有torch_C这个库：主要是因为下载的torch是没有gpu版本的，本项目在训练的时候内存只有60G，无法完全承载训练要求，因此只能每一轮epoch训练以后清空gpu内存。不用gpu训练的话会很慢很慢。gpu训练整套流程花了大概40h？可恶！因此需要gpu。
因为用的monai和python兼容的原因。用到的版本是monai1.5和python3.10，对应的torch是2.7.1+cuda11.8，但是我在测试的时候好像直接install requirements的时候下载不了？？所以直接手动下载吧，去官网里找对应的老版本然后下到本地。其余的包应该可以下。如果pip install无法生效就用conda install吧。






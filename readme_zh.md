````markdown
## [CVPR 2021] Spk2ImgNet：从连续脉冲流重建动态场景的学习

<h4 align="center">赵晶，熊瑞勤，刘航凡，张健，黄铁军</h4>

本仓库包含我们论文的官方源码：

Spk2ImgNet: Learning to Reconstruct Dynamic Scene from Continuous Spike Stream. CVPR 2021

论文：
[Spk2ImgNet-CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhao_Spk2ImgNet_Learning_To_Reconstruct_Dynamic_Scene_From_Continuous_Spike_Stream_CVPR_2021_paper.pdf)

* [Spk2ImgNet](#Learning-to-Reconstruct-Dynamic-Scene-from-Continuous-Spike-Stream.)
  * [运行环境](#Environments)
  * [下载预训练模型](#Download-the-pretrained-models)
  * [评估](#Evaluate)
  * [训练](#Train)
  * [引用](#Citations)


## 运行环境

你需要为你的计算环境选择合适的 cudatoolkit 版本。代码在 PyTorch 1.10.2+cu113 和 spatial-correlation-sampler 0.3.0 上进行了测试，但其他版本也可能可用。

```bash
conda create -n steflow python==3.9
conda activate steflow
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip3 install matplotlib opencv-python h5py
```

我们并不能保证所有 PyTorch 版本都能正常工作。

## 准备数据

### 下载预训练模型

预训练模型可以从下面的 Google Drive 链接下载：

[预训练模型链接](https://drive.google.com/file/d/1vBTJxlctk4otQKsyRq7lsFYGU4WGRNjt/view?usp=sharing)

你可以将预训练模型下载到 `./ckpt`。

### 下载训练数据

训练数据可以从下面的 Google Drive 链接下载：

[训练数据链接](https://drive.google.com/file/d/1ozR2-fNmU10gA_TCYUfJN-ahV6e_8Ke7/view?usp=sharing)

## 评估

你可以在 .py 文件中设置数据路径，或通过命令行参数（--data）传入。

```bash
python3 main_steflow_dt1.py \
--test_data 'Spk2ImgNet_test2' \
--model_name 'model_061.pth'
```


## 训练

所有用于超参数调整的命令行参数都可以在 `train.py` 文件中找到。
你可以在 .py 文件中设置数据路径，或通过命令行参数（--data）传入。

```bash
python3 train.py
```

## 引用

如果你在研究中使用了这份代码，请考虑引用我们的论文：

```
@inproceedings{zhao2021spike,
  title={Spk2ImgNet: Learning to Reconstruct Dynamic Scene from Continuous Spike Stream},
  author={Zhao, Jing and Xiong, Ruiqin and Liu, Hangfan and Zhang, Jian and Huang, Tiejun},
  booktitle={2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021}
}
```



````
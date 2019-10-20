# Lightweight-CNN-FC
This is the Keras implementation of Lightweight-CNN-FC using functional API. The Lightweight-CNN-FC model has the advantages including much smaller model size than that of conventional CNN models while maintaining high accuracy in traffic-sign and vehicle classification.

## Model Layout
![Layout of Lightweight-CNN-FC](https://github.com/zhoujunhao/lightweight-CNN-FC/blob/master/fig/f1.PNG)

This stacked convolutional structure consists of factorization convolutional layers alternating with compression layers.

## Dataset
**GTSRB** dataset contains more than 50,000 traffic sign images, which have been categorized into 40 classes. We select three major categories: Speed-limit signs, Direction signs and Attention signs.

<p align="center">
  <img width="550" height="250" src="https://github.com/zhoujunhao/lightweight-CNN-FC/blob/master/fig/dataset1.PNG">
</p>

**VCifar-100** dataset contains 5 classes: bicycles, buses, motorcycles, pickup trucks, trains.
<p align="center">
  <img width="330" height="80" src="https://github.com/zhoujunhao/lightweight-CNN-FC/blob/master/fig/dataset2.PNG">
</p>

## Experimental Environment

This repository performs the experiments on two platforms: a PC and an MEC platform (i.e., Jetson TX2 module).

**PC Platform**
- Processor: Intel Core i7-7700HQ
- Memory (RAM): 16 GB
- Graphics: NVIDIA GTX 1050
- OS: Ubuntu 16.04

**MEC: Jetson TX2 Module**
- Processor: ARM Cortex-A57 + NVIDIA Denver2
- Memory (RAM): 8 GB
- Graphics: NVIDIA 256-core Pascal
- OS: Ubuntu 16.04

<p align="center">
  <img width="350" height="250" src="https://github.com/zhoujunhao/lightweight-CNN-FC/blob/master/fig/mec.PNG">
</p>

## Installation

- Python 2.7
- Tensorflow-gpu 1.5.0
- Keras 2.1.3
- scikit-learn 0.19

## Train the model

**Run command below to train the model:**
Train Lightweight-CNN-FC model based on GTSRB data.
```
python Lightweight-CNN-FC.py
```
Train the baseline models. For example, you can train VGG16 model as the baseline model.
```
python baseline_vgg16.py
```

# Lightweight-CNN-FC
This is the Keras implementation of Lightweight-CNN-FC using functional API. The Lightweight-CNN-FC model has the advantages including much smaller model size than that of conventional CNN models while maintaining high accuracy in traffic-sign and vehicle classification.

## Model Layout
![Layout of Lightweight-CNN-FC](https://github.com/zhoujunhao/lightweight-CNN-FC/blob/master/fig/f1.PNG)

This stacked convolutional structure consists of factorization convolutional layers alternating with compression layers.

## Dataset
GTSRB dataset contains more than 50,000 traffic sign images, which have been categorized into 40 classes. We select three major categories: Speed-limit signs, Direction signs and Attention signs.

<p align="center">
  <img width="300" height="80" src="https://github.com/zhoujunhao/lightweight-CNN-FC/blob/master/fig/dataset1.PNG">
</p>

<p align="center">
  <img width="300" height="80" src="https://github.com/zhoujunhao/lightweight-CNN-FC/blob/master/fig/dataset2.PNG">
</p>


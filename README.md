# [Adas: Adaptive Scheduling of Stochastic Gradients](https://openreview.net/forum?id=qUzxZj13RWY) #
## Status ##
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![maintenance](https://img.shields.io/badge/maintained%3F-yes-brightgreen.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![python](https://img.shields.io/badge/python-v3.7-blue)](https://www.python.org/downloads/release/python-370/)
![size](https://img.shields.io/github/repo-size/mahdihosseini/AdaS)

## Table of Contents ##
- [Adas: Adaptive Scheduling of Stochastic Gradients](#adas--adaptive-scheduling-of-stochastic-gradients)
  * [Status](#status)
  * [Introduction](#introduction)
    + [License](#license)
    + [Citing AdaS](#citing-adas)
    + [Empirical Classification Results on CIFAR10 and CIFAR100](#empirical-classification-results-on-cifar10-and-cifar100)
    + [QC Metrics](#qc-metrics)
  * [Requirements](#requirements)
    + [Software/Hardware](#software-hardware)
    + [Computational Overhead](#computational-overhead)
    + [Installation](#installation)
    + [Usage](#usage)
    + [Common Issues (running list)](#common-issues--running-list-)
  * [TODO](#todo)
    + [Pytest](#pytest)

## Introduction ##
**[AdaS](https://openreview.net/forum?id=qUzxZj13RWY)** is an adaptive optimizer for scheduling the learning rate in training Convolutional Neural Networks (CNN)

- AdaS exhibits the rapid minimization characteristics that adaptive optimizers like [AdaM](https://arxiv.org/abs/1412.6980) are favoured for
- AdaS exhibits *generalization* (low testing loss) characteristics on par with SGD based optimizers, improving on the poor *generalization* characteristics of adaptive optimizers
- AdaS introduces no computational overhead over adaptive optimizers (see [experimental results](#some-experimental-results))
- In addition to optimization, AdaS introduces new probing metrics for CNN layer evaulation ([quality metrics](#knowledge-gain-vs-mapping-condition---cnn-quality-metrics))

This repository contains a [PyTorch](https://pytorch.org/) implementation of the AdaS learning rate scheduler algorithm.

### License ###
AdaS is released under the MIT License (refer to the [LICENSE](LICENSE) file for more information)
|Permissions|Conditions|Limitations|
|---|---|---|
|![license](https://img.shields.io/badge/-%20-brightgreen) Commerical use|![license](https://img.shields.io/badge/-%20-blue) License and Copyright Notice|![license](https://img.shields.io/badge/-%20-red) Liability|
|![license](https://img.shields.io/badge/-%20-brightgreen) Distribution| | ![license](https://img.shields.io/badge/-%20-red) Warranty|
|![license](https://img.shields.io/badge/-%20-brightgreen) Modification | | |
|![license](https://img.shields.io/badge/-%20-brightgreen) Private Use| | |

### Citing AdaS ###
```text
@article{hosseini2020adas,
  title={AdaS: Adaptive Scheduling of Stochastic Gradients},
  author={Hosseini, Mahdi S and Plataniotis, Konstantinos N},
  journal={arXiv preprint arXiv:2006.06587},
  year={2020}
}
```
### Empirical Classification Results on [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html), [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html) and [Tiny-ImageNet-200](http://cs231n.stanford.edu/tiny-imagenet-200.zip) ###

**Figure 1: Training performance using different optimizers across three datasets and two CNNs**
![figure 1](figures/main_results.png)


**Table 1: Image classification performance (test accuracy) with fixed budget epoch of ResNet34 training**
![table 1](figures/tabular_results.png)

### QC Metrics ###
Please refer to [QC on Wiki](https://github.com/mahdihosseini/AdaS/wiki/On-Quality-Metrics) for more information on two metrics of knowledge gain and mapping condition for monitoring training quality of CNNs

## Requirements ##
### Software/Hardware ###
We use `Python 3.7`.

Please refer to [Requirements on Wiki](https://github.com/mahdihosseini/AdaS/wiki/On-Installation-Requirements) for complete guideline.

### Computational Overhead ###
AdaS introduces no overhead (very minimal) over adaptive optimizers e.g. all mSGD+StepLR, mSGD+AdaS, AdaM consume 40~43 sec/epoch to train ResNet34/CIFAR10 using the same PC/GPU platform

### Installation ###
There are two versions of the AdaS code contained in this repository.
1. a python-package version of the AdaS code, which can be `pip`-installed.
2. a static python module (unpackaged), runable as a script.

All source code can be found in [src/adas](src/adas)

For more information, also refer to [Installation on Wiki](https://github.com/mahdihosseini/AdaS/wiki/On-Package-Installation)


### Usage ###
Moving forward, I will refer to console usage of this library. IDE usage is no different. Training options are split two ways:
1. all environment/infrastructure options (GPU usage, output paths, etc.) is specified using arguments.
2. training specific options (network, dataset, hyper-parameters, etc.) is specified using a configuration **config.yaml** file:

```yaml
###### Application Specific ######
dataset: 'CIFAR10'
network: 'VGG16'
optimizer: 'SGD'
scheduler: 'AdaS'


###### Suggested Tune ######
init_lr: 0.03
early_stop_threshold: 0.001
optimizer_kwargs:
  momentum: 0.9
  weight_decay: 5e-4
scheduler_kwargs:
  beta: 0.8

###### Suggested Default ######
n_trials: 5
max_epoch: 150
num_workers: 4
early_stop_patience: 10
mini_batch_size: 128
p: 1 # options: 1, 2.
loss: 'cross_entropy'
```

For complete instruction on configuration and different parameter setup, please refer to [Configuration on Wiki](https://github.com/mahdihosseini/AdaS/wiki/On-Configuration-File)

### Common Issues (running list) ###
- None :)

## TODO ###
- Add medical imaging datasets (e.g. digital pathology, xray, and ct scans)
- Extension of AdaS to Deep Neural Networks

### Pytest ###
Note the following:
- Our Pytests write/download data/files etc. to `/tmp`, so if you don't have a `/tmp` folder (i.e. you're on Windows), then correct this if you wish to run the tests yourself

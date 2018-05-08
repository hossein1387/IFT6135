# ICLR 2018 Reproducibility Challenge:

We reproduced 'Training and Inference with Integers in Deep Neural Networks' in PyTorch

## Abstract

We reproduce Wu et al.'s ICLR 2018 submission *Training And Inference With Integers In Deep Neural Networks*.
The proposed `WAGE' model reduces floating-point precision with only a slight reduction in accuracy.
The paper introduces two novel approaches which allow for the use of integer values by quantizing weights, activation, gradients and errors in both training and inference.
We reproduce the WAGE model, trained on the CIFAR10 dataset. The methodology demonstrated in this paper has applications for use with Application Specific Integrated Circuit (ASICs).

## Requirements

Python3

## Installation

    git clone https://github.com/hossein1387/IFT6135.git
    cd IFT6135/Project
    pip3 install -r 'requirements.txt'

## Usage

* Download CIFAR10 or MNIST dataset
* `mv YOUR_DATASET Project/src/data/raw/YOUR_DATASET`
* Set hyperparameters in `config.yaml`
* Run with `python main.py`

## Authors
* Hossein Akskari
* Isaac Sultan
* Breandan Considine

## Acknowledgments

* Wu et Al. ['Training and Inference with Integers in Deep Neural Networks'](https://arxiv.org/abs/1802.04680)


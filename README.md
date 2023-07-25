# Bidirectional Looking with A Novel Double Exponential Moving Average to Adaptive and Non-adaptive Momentum Optimizers
This project focuses on the optimizer framework Admeta, which has a bidirectional view with forward-looking and backward-looking. For practice, we implement our framework based on RAdam and SGDM and thus propose AdmetaR and AdmetaS. 

## Installation via pip

## Paper link
https://proceedings.mlr.press/v202/chen23r/chen23r.pdf
## How to cite this work
```text
@inproceedings{chen2023bidirectional,
  title={Bidirectional Looking with A Novel Double Exponential Moving Average to Adaptive and Non-adaptive Momentum Optimizers},
  author={Chen, Yineng and Li, Zuchao and Zhang, Lefei and Du, Bo and Zhao, Hai},
  booktitle={International Conference on Machine Learning},
  pages={4764--4803},
  year={2023},
  organization={PMLR}
}
```
## Quick introduction to this code
As can be seen in the paper, we conduct experiments to test the performance of AdmetaR and AdmetaS on several tasks. 

For Image Classification (cifar10 and cifar100), please go to admeta-code/admeta-cifar/pytorch_image_classification.
For results in GLUE benchmark, please go to admeta-code/admeta-transformers/admeta-transformers/examples/pytorch/text-classification.
For machine reading comprehension and named entity recognition, please go to admeta-code/admeta-transformers/admeta-transformers/examples/pytorch/question-answering and admeta-code/admeta-transformers/admeta-transformers/examples/pytorch/token-classification.
For audio classification, please go to admeta-code/admeta-transformers/admeta-transformers/examples/pytorch/audio-classification.

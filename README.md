# Bidirectional Looking with A Novel Double Exponential Moving Average to Adaptive and Non-adaptive Momentum Optimizers
Optimizer is an essential component for the success of deep learning, which guides the neural network to update the parameters according to the loss on the training set. SGD and Adam are two classical and effective optimizers on which researchers have proposed many variants, such as SGDM and RAdam. In this paper, we innovatively combine the backward-looking and forward-looking aspects of the optimizer algorithm and propose a novel \textsc{Admeta} (\textbf{A} \textbf{D}ouble exponential \textbf{M}oving averag\textbf{E} \textbf{T}o \textbf{A}daptive and non-adaptive momentum) optimizer framework. For backward-looking part, we propose a DEMA variant scheme, which is motivated by a metric in the stock market, to replace the common exponential moving average scheme. While in the forward-looking part, we present a dynamic lookahead strategy which asymptotically approaches a set value, maintaining its speed at early stage and high convergence performance at final stage. Based on this idea, we provide two optimizer implementations, \textsc{AdmetaR} and \textsc{AdmetaS}, the former based on RAdam and the latter based on SGDM. Through extensive experiments on diverse tasks, we find that the proposed \textsc{Admeta} optimizer outperforms our base optimizers and shows advantages over recently proposed competitive optimizers. We also provide theoretical proof of these two algorithms, which verifies the convergence of our proposed \textsc{Admeta}. 


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

- For `Image Classification (cifar10 and cifar100)`, please go to `admeta-code/admeta-cifar/pytorch_image_classification`.
  
- For `GLUE benchmark`, please go to `admeta-code/admeta-transformers/admeta-transformers/examples/pytorch/text-classification`.
  
- For `machine reading comprehension and named entity recognition`, please go to `admeta-code/admeta-transformers/admeta-transformers/examples/pytorch/question-answering` and `admeta-code/admeta-transformers/admeta-transformers/examples/pytorch/token-classification`.
  
- For `audio classification`, please go to `admeta-code/admeta-transformers/admeta-transformers/examples/pytorch/audio-classification`.

The code of the optimizers are already integrated in `admeta-code/admeta-cifar/pytorch_image_classification/optim/` and `admeta-code/admeta-transformers/admeta-transformers/src/transformers/optimization.py`

## Algorithms
<p align='center'>
<img src="Images/algorithm 1.jpg" width="50%"><img src="Images/algorithm 2.jpg" width="50%"> 
</p>


# Q&As for the warm up section

## General Machine Learning
1. **Q**: What is necessary for a machine learning task to be considered correctly posed?  
   **A**: We need several main components, each of which is essential to consider the machine learning task to be considered
   correctly posed:
    1) Training set: pairs of objects - labels.
    2) Parametric family of algorithms from will we will be selecting a
       model by fine-tuning parameters (training process). For instance, we can decide that we will be training a
       linear model or a neural network of a particular architecture.
       Linear models and neural networks are a parametric family of algorithms.
    3) Objective (loss function) - the actual goal that defines what
       kind of algorithms we are looking for.
    4) Metric - measure to define the quality of the trained algorithm.


2. **Q**: What is overfitting, and how to detect it?  
   **A**: Overfitting is the production of an analysis that corresponds too closely or exactly to a particular set of 
   data and may therefore fail to fit additional data or predict future observations reliably. 
   Typically, overfitting is detected by performing validation (verification of the trained model's quality on a 
   separate dataset) and measuring the value of validation loss. During the model fitting, train and validation 
   losses move towards each other while overfitting typically cause their divergence. There are other techniques for 
   overfitting detection, such as analysis of the model's parameters.


3. **Q**: What is model regularization? What kinds of regularization are you familiar with?  
   **A**: Regularization is a technique that makes slight modifications to the learning algorithm such that the model 
   generalizes better. This, in turn, is supposed to improve the model's performance on the unseen data. In other words,
   it is a technique to deal with overfitting. There are plenty of regularization methods. Some of them are:
   L1/L2 weights regularization, dropout (for NNs), batch normalization (here regularization is a side effect),
   label smoothing and entropy regularization for the classification task, etc.

## General Deep Learning

1. **Q**: What optimizers do you know? What is the difference between them? How do they work?  
   **A**: GD, SGD, SGD-momentum, Nesterov SGD-momentum, RMSProp, Adam, Adam are the most common,
   The main high-level difference between them is that starting from SGD, we add terms to the optimization algorithms,
   which helps us to reduce gradient oscillations caused by the stochastic nature of the optimization process.
   SGD-momentum and Nesterov SGD-momentum add somewhat like a moving average to the result to smooth gradient values,
   RMSProp uses an adaptive learning rate instead of treating the learning rate as a hyperparameter.
   This means that the learning rate changes over time. Adam is a combination of thereof. Adam introduces a term to
   rectify the variance of the adaptive learning rate, which arguably is the root cause of sometimes bad convergence
   of Adam.


2. **Q**: How to debug neural networks?  
   **A**: No simple answer to that. The answer can vastly vary from printing out stuff to in-depth analysis of 
   network's weighs gradients etc. This question is intended to show whether the candidate has faced real problems 
   with complicated models.

## Deep Learning in Computer Vision

1. **Q**: Please name current SOTA approaches for typical DL tasks such as classification, detection, segmentation
   (may be other tasks) on general domain.  
   **A**: As for Jan 14'th 2021 SOTA are:
    - Classification - [Meta Pseudo labels + EfficientNet V2](https://arxiv.org/abs/2003.10580) - 90.2% accuracy
      on ImageNet 1k.
    - Object detection - [Cascade Eff-B7 NAS-FPN](https://arxiv.org/pdf/2012.07177v1.pdf) - 57.3 Box AP on COCO.
    - Binary semantic segmentation - [EfficientNet-L2+NAS-FPN](https://arxiv.org/abs/2006.06882v2) - 90.5% Mean IOU on
      PASCAL VOC 2012.


2. **Q**: What is UNet, and how it works? Why does it need to skip connections?  
   **A**: UNet is a CV model architecture initially designed for binary semantic segmentation of biomedical data.
   There are two major design decisions in UNet: encoder-decoder architecture and skip connections between encoder 
   and decoder blocks. The first decision is made for efficiency (fewer parameters), the second one to prevent overfitting:
   gradients do not go to deeper layers if they are not needed.


3. **Q**: What approaches to face recognition and searching for similar objects are you familiar with? How do they work?  
   **A**: [Siamese Nets](https://medium.com/@enoshshr/triplet-loss-and-siamese-neural-networks-5d363fdeba9b),
   [Margin Loss](https://papers.nips.cc/paper/2003/file/0fe473396242072e84af286632d3f0ff-Paper.pdf),
   [Triplet Loss](https://medium.com/@enoshshr/triplet-loss-and-siamese-neural-networks-5d363fdeba9b),
   [ArcFace](https://arxiv.org/abs/1801.07698)
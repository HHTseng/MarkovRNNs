# Markov Recurrent Neural Networks

This repository is the PyTorch implementation of [Markov Recurrent Neural Networks](https://github.com/NCTUMLlab/Che-Yu-Kuo-MarkovRNN.git) with two temporal datasets as quick demonstration.

## Architecture
**Paper**: [Markov Recurrent Neural Networks](https://ieeexplore.ieee.org/document/8517074)

<img src="./fig/NTMCell.png" width="100%">

## Dataset

1. [MNIST](https://en.wikipedia.org/wiki/MNIST_database) viewed in series as sequential input.


<img src="./fig/image3.png" width="100%">


2. Artificial alien signals: I am imagining we are able to recognize radio signals sent by aliens from the sky [SETI](https://setiathome.berkeley.edu/), where I generated two kinds of wave forms for distinguish two patterns:

<img src="./fig/alien_wave.png" width="100%">


### Prerequisites
- [Python 3.6](https://www.python.org/)
- [Jupyter notebook](https://jupyter.org/)
- [PyTorch 1.0](https://pytorch.org/)
- [Numpy 1.15.0](http://www.numpy.org/)
- [Sklearn 0.20.2](https://scikit-learn.org/stable/)
- [Matplotlib](https://matplotlib.org/)
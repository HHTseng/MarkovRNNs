# Markov Recurrent Neural Networks

This repository is the PyTorch implementation of [Markov Recurrent Neural Networks](https://github.com/NCTUMLlab/Che-Yu-Kuo-MarkovRNN.git) with two temporal datasets as quick demonstration.

## Architecture
- **Paper**: [Markov Recurrent Neural Networks (MRNN)](https://ieeexplore.ieee.org/document/8517074)

<img src="./fig/NTMCell.png" width="100%">


- **Heuristic explanation:**
MRNN is built as a deep learning model for time series, such as NLP, stock price prediction, or gravitational wave detection. The main idea is to create several *parallel* [RNNs](https://en.wikipedia.org/wiki/Recurrent_neural_network) [(LSTMs)](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) to learn the time dependence of the data simultaneously. If data has complex temporal structures (behaviour), single RNN may not be enough to carry out the pattern. *k* parallel RNNs (*k=1,2,3,...*) can read same input signal at the same time, each learns different character of data. Then another latent variable *z* (also trained by networks) will determine when and which LSTM should be listened for attaining learning task (see Fig *q_z(t)* & *z(t)* below). The choosing mechanism by *z* itself is a process stochastic modeling of transitions between *k* LSTMs based on Markov property, and hence the name MRNN.

- **Note:**
The transition variable z between *k* LSTMs can regarded as an [attention mechanism](https://nlp.stanford.edu/pubs/emnlp15_attn.pdf) over individual LSTM hidden states.


## Datasets

- [**MNIST**](https://en.wikipedia.org/wiki/MNIST_database) viewed in series as sequential input:
<img src="./fig/image3.png" width="100%">

- **Artificial alien signals**: I am imagining we are able to recognize radio signals sent by aliens from the sky such as [SETI](https://setiathome.berkeley.edu/), where I generated two kinds of wave forms for Markov RNN to distinguish:

<img src="./fig/alien_wave.png" width="100%">

## Prerequisites
- [Python 3.6](https://www.python.org/)
- [Jupyter notebook](https://jupyter.org/)
- [PyTorch 1.0](https://pytorch.org/)
- [Numpy 1.15.0](http://www.numpy.org/)
- [Sklearn 0.20.2](https://scikit-learn.org/stable/)
- [Matplotlib](https://matplotlib.org/)


## Usage
No installation required except the prerequisites. File `kLSTM.py` contains all the modules needed for running Markov RNN. Two examples are provided in Jupyter notebook formats:
```
MRNN_MNIST.ipynb
MRNN_detect_alien_signal.ipynb
```

## Results & Interpretations
1. **Take *k=4* LSTM for MNIST.**

- In figure of *q_z(t)*, the horizontal axis is time, vertical axis shows which LSTM to look. The color palette indicates the probability of the LSTM being used. 

- In figure of *z(t)*, the yellow color indicates which LSTM was actually used (by Gumbel softmax sampling).
<img src="./fig/qz_t_digit3.png" width="100%">

- Over all probability of which LSTM being chosen. **[Left]: digit 7**, **[right]: digit 9**
<img src="./fig/digit_transitions.png" width="100%">


2. **Take *k=4* LSTM for alien signal (binary) classification**
- Non-alien signal
- Fig *q_z(t)* & *z(t)* slightly shows periodicity
<img src="./fig/qz_t_wave0.png" width="100%">


- Alien signal (maybe say Hi or invasion)
- Fig *q_z(t)* & *z(t)* use 4 LSTMs to detect irregular wave form from aliens.
<img src="./fig/qz_t_wave1.png" width="100%">

- Over all probability of which LSTM being chosen. **[Left]: non-alien signal**, **[right]: alien signal**
<img src="./fig/wave_transitions.png" width="100%">



## Improvements
This code of Pytorch is extended such that the Markov RNN can have more than 1 hidden layers in every LSTM, where it was restricted to only 1 hidden layer in the original code of [Tensorflow version](https://github.com/NCTUMLlab/Che-Yu-Kuo-MarkovRNN.git).

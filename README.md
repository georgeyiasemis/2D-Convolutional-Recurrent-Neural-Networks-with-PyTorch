# 2D Convolutional Recurrent Neural Networks with PyTorch
## Two dimensional Convolutional Recurrent Neural Networks implemented in PyTorch

The architecture of ```Conv2dLSTMCell``` was inspired by "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting" 
(https://arxiv.org/pdf/1506.04214.pdf).

See the image below for the key equations of ```Conv2dLSTMCell```:

![Capture](https://user-images.githubusercontent.com/71031687/112730543-73de0900-8f32-11eb-8396-a79091979335.JPG)


The implementations of ```Conv2dRNNCell``` and ```Conv2dGRUCell``` are  based on the implementation of Convolutional LSTM.


This repo contains implementations of:

  * Conv2dRNNCell
  * Conv2dLSTMCell 
  * Conv2dGRUCell
  
and

  * Conv2dRNN / Biderectional Conv2dRNN
  * Conv2dLSTM / Biderectional Conv2dLSTM
  * Conv2dGRU / Biderectional Conv2dGRU.

## Dependencies

* ```pytorch```
* ```numpy```

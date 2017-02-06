
# Introduction
Sophia (Greek for "wisdom") is a real-time recurrent neural network (RNN)
agent based on Theano. It focuses on training and evaluating experimental RNN
architectures for regression tasks on long and noisy inputs.

## Features
* [BPTT(h; h')](https://doi.org/10.1162/neco.1990.2.4.490)
  for more efficient training
* On-line, minibatch training on sequences of unequal lengths
* Inference on ensemble of models
* Scheduled learning rate annealing with patience
* LSTM/GRU with various unit options
 ([weight normalization](https://arxiv.org/abs/1602.07868),
  [residual gate](https://arxiv.org/abs/1611.01260),
  [initial state learning](https://www.cs.toronto.edu/~hinton/csc2535/notes/lec10new.pdf),
  etc.), loss functions
 (L2, L1, [Huber](https://en.wikipedia.org/wiki/Huber_loss)),
  and optimizers
 ([Nesterov](https://arxiv.org/abs/1212.0901),
  [Adadelta](https://arxiv.org/abs/1212.5701),
  [RMSProp](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf),
  [Adam](https://arxiv.org/abs/1412.6980))
* Options for learning time-dimension
  (e.g., [Phased LSTM](https://arxiv.org/abs/1610.09513))
  or batch-dimension
  (e.g., [sequence ID](https://doi.org/10.1109/ICASSP.2013.6639211))
  parameters
* Option for faster training by unrolling Theano scans (increases compile time)
* Self-documenting object oriented code

## Requirements
* Python 2.7 or 3.x
* Theano (developed on 0.9.0.dev4-py2.7); see ../etc/theano_install_notes.txt 
* ZeroMQ : only used for communicating real-time input/output data with an
  external process
* Input and target data preprocessed and saved as separate binary files
  ([timesteps][dimensions] order)

# How to use

## Options
|     option name      |               explanation               |
|:--------------------:|:---------------------------------------:|
|input_dim, target_dim |integers                                 |
|      unit_type       |'fc'/'lstm'/'gru'                        |
|      loss_type       |'l2'/'l1'/'huber'                        |
| net_width, net_depth |# of params ~ W<sup>2</sup> D            |
|      batch_size      |minibatch size                           |
|window_size, step_size|for BPTT(window_size; step_size)         |
|     weight_norm      |False/True                               |
|    residual_gate     |False/True                               |
|  learn_init_states   |False/True                               |
|  learn_id_embedding  |False/True                               |
|  learn_clock_params  |False/True                               |
|     update_type      |'sgd'/'momentum'/'nesterov'              |
|      force_type      |'vanilla'/'adadelta'/'rmsprop'/'adam'    |
|   frames_per_epoch   |time_indices * batch_size per epoch      |
|   lr_*, max_retry    |for learning rate annealing with patience|
|     unroll_scan      |trades memory consumption & slower compile time for faster training|

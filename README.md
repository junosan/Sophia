
[//]: # "=========================================================================="
[//]: # " Copyright (C) 2016 Hosang Yoon (hosangy@gmail.com) - All Rights Reserved "
[//]: # " Unauthorized copying of this file, via any medium is strictly prohibited "
[//]: # "                       Proprietary and confidential                       "
[//]: # "=========================================================================="


# Introduction
Sophia (Greek for "wisdom") is based on Theano and focuses on testing 
experimental recurrent neural network architectures for regression tasks on
long and noisy inputs.

Notable features:
* Implements [BPTT(h; h')](https://doi.org/10.1162/neco.1990.2.4.490)
  for more efficient training
* Scheduled learning rate annealing with patience
* Various unit options
 ([batch norm](https://arxiv.org/abs/1603.09025),
  [residual gate](https://arxiv.org/abs/1611.01260),
  initial state learning, etc.)
  and optimizers
 ([Nesterov](https://arxiv.org/abs/1212.0901),
  [Adadelta](https://arxiv.org/abs/1212.5701),
  [RMSProp](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf),
  [Adam](https://arxiv.org/abs/1412.6980))
* Options for learning time-dimension
  (e.g., [Phased LSTM](https://arxiv.org/abs/1610.09513))
  or batch-dimension (e.g., sequence ID) parameters
* Option for faster training by unrolling Theano scans (increases compile time)
* Self-documenting object oriented code


# How to use

## Requirements
* Theano (developed on 0.9.0.dev4-py2.7): follow through the
  [installation instructions](http://deeplearning.net/software/theano_versions/dev/install.html)
  using Miniconda, along with the bleeding-edge installation of Theano,
  latest version of libgpuarray, and cuDNN
* Input and target data preprocessed and saved as separate binary files
  ([timesteps][dimensions] order)

## Options
|     option name      |                            explanation                            |
|:--------------------:|:-----------------------------------------------------------------:|
|input_dim, target_dim |currently set to 44, 1                                             |
|      unit_type       |'FC'/'LSTM'/'GRU'                                                  |
| net_width, net_depth |# of params ~ W<sup>2</sup> D                                      |
|      batch_size      |minibatch size                                                     |
|window_size, step_size|for [BPTT(h; h')](doi:10.1162/neco.1990.2.4.490)                   |
|      batch_norm      |False/[True](https://arxiv.org/abs/1603.09025)                     |
|    residual_gate     |False/[True](https://arxiv.org/abs/1611.01260)                     |
|  learn_init_states   |False/True                                                         |
|  learn_clock_params  |False/[True](https://arxiv.org/abs/1610.09513)                     |
|     update_type      |'sgd'/'momentum'/'nesterov'                                        |
|      force_type      |'vanilla'/'adadelta'/'rmsprop'/'adam'                              |
|   frames_per_epoch   |time_indices * batch_size per epoch                                |
|   lr_*, max_retry    |for learning rate annealing with patience                          |
|     unroll_scan      |trades memory consumption & slower compile time for faster training|


# Code reference

## Naming convention 

|  prefix   |      type       |                                   explanation                                    |                          how it's created                           |
|:---------:|:---------------:|:--------------------------------------------------------------------------------:|:-------------------------------------------------------------------:|
|  (none)   | built-in/NumPy  |variable on CPU memory; the usual stuff                                           |the usual way                                                        |
|    v_     |th.SharedVariable|variable on GPU memory; has init value                                            |th.shared                                                            |
|    s_     |  symbolic node  |graph node; has no substance until compiled                                       |tensor literals (tt.alloc, etc)   <br> calculations on v_/s_         |
|port_<i/o>_|  symbolic node  |same as above, but marked for use as       <br> input/output ports for th.function|inputs: tt.scalar, tt.matrix, etc <br> outputs: calculations on v_/s_|
|    f_     | callable object |links built-in/NumPy to input/output ports <br> all v_ updates take place via f_  |th.function                                                          |

* `[optional], <required>, {default}, (miscellaneous)`
* np = numpy, th = theano, tt = theano.tensor
* Prefix not used inside Layer.setup_graph as there is no need for distinction there

<br>


|           suffix            |              explanation              |        
|:---------------------------:|:-------------------------------------:|
|<tensor_name>_[t][b][i/j/k/l]|[time][batch][hidden] dimensions       |
|     <individual_name>s      |collections (list, dict, etc)          |

<br>


## Summary of collections in class Net

|    name     |   type    |        element        |                                          explanation                                          |
|:-----------:|:---------:|:---------------------:|:---------------------------------------------------------------------------------------------:|
|   options   |OrderedDict|     str : varies      |training options; saved along with _params                                                     |
|   params    |OrderedDict|   str : np.ndarray    |learnable parameters (on CPU memory)             <br> used for initialization or as a buffer   |
|  v_params   |OrderedDict|str : th.SharedVariable|learnable parameters (on GPU memory)                                                           |
|   v_grads   |    list   |   th.SharedVariable   |loss gradients wrt above; same order as above                                                  |
|v_prev_states|OrderedDict|str : th.SharedVariable|state values right before this time step starts                                                |
|  *_updates  |    list   |       (v_$, s_$)      |give this to th.function to make a function that <br> actually performs the updates when called|

<br>


## Summary of functions in class Net

|            name              |                                    explanation; input -> output                                   |
|:----------------------------:|:-------------------------------------------------------------------------------------------------:|
|         _init_params         |add layers and fill params (either freshly, or from file)                                          |
|    _init_shared_variables    |fill v_params[, v_grads], v_prev_states                                                            |
|     _setup_forward_graph     |s_input, s_time, s_id_idx, s_next_prev_idx -> s_output, prev_state_updates, stat_updates           |
|    _setup_inference_graph    |setup full graph and store ports ([input, time, id_idx], [output]) and updates (prev_state_updates)|
|      _setup_loss_graph       |s_output, s_target, s_step_size -> s_loss                                                          |
|  _setup_grad_updates_graph   |s_loss, v_wrt, v_grads -> grad_updates                                                             |
|  _setup_param_updates_graph  |s_lr, v_params, v_grads -> optim_state_inits, param_updates                                        |
|    _setup_training_graph     |setup full graph and store ports ([input, target, time, id_idx, step_size, lr], [loss]) and updates (prev_state_updates, stat_updates, grad_updates, optim_state_inits, param_updates)|
|   compile_f_fwd_propagate    |(training): _ -> th.function([input, target, time, id_idx, step_size], [loss], prev_state_updates) <br> (inference): _ -> th.function([input, time, id_idx], [output], prev_state_updates)|
| compile_f_fwd_bwd_propagate  |_ -> th.function([input, target, time, id_idx, step_size], [loss], grad_updates + prev_state_updates + stat_updates)|
|  compile_f_update_v_params   |_ -> th.function([lr], [], param_updates)                                                          |
|compile_f_initialize_optimizer|_ -> th.function([], [], optim_state_inits)                                                        |

* *target* means ground truth from files, *output* means what comes out of
  the net
* All updates to shared variables take place via repeated calls to f_*
  (callable objects returned by th.function)
* Within a single update, the order between elements in the updates list
  [does not matter](http://stackoverflow.com/questions/28205589/the-update-order-of-theano-functions-update-list)

<br>


## TODO
- Reinforcement learning on pre-trained nets

## Miscellaneous notes
* [NumPy/Theano broadcasting rules](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
* [SliceableOrderedDict](http://stackoverflow.com/questions/30975339/slicing-a-python-ordereddict)


# Acknowledgements
Consulted code from
* [Theano LSTM implementation reference](http://deeplearning.net/tutorial/lstm.html) (license not specified)
* [Learning rate annealing with patience](https://github.com/KyuyeonHwang/Fractal) (Apache-2.0 license)

[//]: # "=========================================================================="
[//]: # " Copyright (C) 2016 Hosang Yoon (hosangy@gmail.com) - All Rights Reserved "
[//]: # " Unauthorized copying of this file, via any medium is strictly prohibited "
[//]: # "                       Proprietary and confidential                       "
[//]: # "=========================================================================="

# Introduction
Sophia (Greek for "wisdom") is based on Theano and focuses on training and evaluating
recurrent neural networks for regression tasks on long and noisy input with a 
loosely correlated target. 


# How to use

## Options
- input_dim, target_dim : 44, 1
- unit_type             : 'LSTM'
- net_width, net_depth  : complexity ~ W<sup>2</sup> D
- batch_size            : 64
- step_size             : 128
- rolling_first_step    : True
- learn_init_states     : False
- layer_norm            : False ([arXiv:1607.06450](https://arxiv.org/abs/1607.06450))
- learn_clock_params    : False ([arXiv:1610.09513](https://arxiv.org/abs/1610.09513))
- clock_t_exp_lo        : 1
- clock_t_exp_hi        : 7
- clock_r_on            : 0.1
- clock_leak_rate       : 0.001
- optimizer             : 'adam'
- grad_clip             : 2.
- lr_init_val           : 0.001
- lr_lower_bound        : 0.0001
- lr_decay_rate         : 0.5
- max_retry             : 10
- frames_per_epoch      : 8 * 1024 * 1024


# Code reference

## Naming convention 

|  prefix   |      type       |                                   explanation                                    |                          how it's created                           |
|:---------:|:---------------:|:--------------------------------------------------------------------------------:|:-------------------------------------------------------------------:|
|  (none)   | built-in/NumPy  |variable on CPU memory; the usual stuff                                           |the usual way                                                        |
|    v_     |th.SharedVariable|variable on GPU memory; has init value                                            |th.shared                                                            |
|    s_     |  symbolic node  |graph node; has no substance until compiled                                       |tensor literals (tt.alloc, etc)   <br> calculations on v_/s_         |
|port_<i/o>_|  symbolic node  |same as above, but marked for use as       <br> input/output ports for th.function|inputs: tt.scalar, tt.matrix, etc <br> outputs: calculations on v_/s_|
|    f_     |callable object  |links built-in/NumPy to input/output ports <br> all v_ updates take place via f_  |th.function                                                          |

* th = theano, tt = theano.tensor
* Prefix not used inside Layer.setup_graph as there is no need for distinction there

<br>


|         suffix          |           explanation          |        
|:-----------------------:|:------------------------------:|
|<tensor_name>_[t][b][i/j]|[time][batch][hidden] dimensions|
|   <individual_name>s    |collections (list, dict, etc)   |

` [optional], <required>, {default}, (miscellaneous) `

<br>


## Summary of collections in class Net

|     name     |   type    |        element        |                                          explanation                                          |
|:------------:|:---------:|:---------------------:|:---------------------------------------------------------------------------------------------:|
|   _options   |OrderedDict|     str : varies      |training options; saved along with _params                                                     |
|   _params    |OrderedDict|   str : np.ndarray    |learnable parameters (on CPU memory)             <br> used for initialization or as a buffer   |
|  _v_params   |OrderedDict|str : th.SharedVariable|learnable parameters (on GPU memory)                                                           |
|   _v_grads   |    list   |   th.SharedVariable   |loss gradients wrt above; same order as above                                                  |
|_v_prev_states|    dict   |str : th.SharedVariable|state values right before this time step         <br> clock and layers own one state each      |
|  _*_updates  |    list   |       (v_$, s_$)      |give this to th.function to make a function that <br> actually performs the updates when called|

<br>


## Summary of functions in class Net

|            name              |                                                           explanation; input -> output                                                            |
|:----------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------:|
|         _init_params         |add layers and fill _params (either freshly, or from file)                                                                                         |
|    _init_shared_variables    |fill _v_params, _v_grads, _v_prev_states                                                                                                           |
|     _setup_forward_graph     |s_input, s_time, v_params, v_prev_stats -> s_output, prev_state_updates                                                                            |
|    _setup_inference_graph    |store _port_i_input, _port_i_time, _port_o_output, _prev_state_updates                                                                             |
|      _setup_loss_graph       |s_output, s_target -> s_loss                                                                                                                       |
|  _setup_grad_updates_graph   |s_loss, v_params, v_grads -> grad_updates                                                                                                          |
|  _setup_param_updates_graph  |s_lr, v_params, v_grads -> optim_state_inits, param_updates                                                                                        |
|    _setup_training_graph     |store _port_i_input, _port_i_target, _port_i_time, _port_i_lr, _port_o_loss, _prev_state_updates, _grad_updates, _optim_state_inits, _param_updates|
| compile_f_initialize_states  |_ -> th.function([], [], updates=_v_prev_states <- 0-tensor or _params)                                                                            |
|   compile_f_fwd_propagate    |_ -> th.function([_port_i_input, _port_i_time], _port_o_output, updates=_prev_state_updates)                                                       |
| compile_f_fwd_bwd_propagate  |_ -> th.function([_port_i_input, _port_i_target, _port_i_time], _port_o_loss, updates=_prev_state_updates, _grad_updates)                          |
|  compile_f_update_v_params   |_ -> th.function(_port_i_lr, [], updates=_param_updates)                                                                                           |
|compile_f_initialize_optimizer|_ -> th.function([], [], updates=_optim_state_inits)                                                                                               |

* *target* means ground truth from files, *output* means what comes out of the net
* All updates to shared variables take place via repeated calls to f_* (callable objects returned by th.function)
* Within a single update, [the order between elements in the updates list does not matter](http://stackoverflow.com/questions/28205589/the-update-order-of-theano-functions-update-list)

<br>


## TODO

- rolling_first_step (1~h)
- Ensemble
- implement learnable clocks - turn on/off with an option?
- test pybind11 (are Python states kept between calls from C++ code)
    if not, use zeroMQ


## Miscellaneous notes
* About launching multiple processes: https://groups.google.com/forum/#!topic/theano-dev/jzB08629Vvw
* [SliceableOrderedDict](http://stackoverflow.com/questions/30975339/slicing-a-python-ordereddict)

# Acknowledgements
Used partial code from:
- Layer normalization implementation: https://github.com/ryankiros/layer-norm (license not specified)
- Weight initializers: https://github.com/ivendrov/order-embedding (Apache License 2.0)
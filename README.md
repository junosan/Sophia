
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

## Prerequisites
- Theano (developed on 0.9.0.dev4-py2.7): follow through their installation instructions using Miniconda, along with the bleeding-edge installation & latest version of libgpuarray
- Input and target data preprocessed and saved as separate binary files ([timesteps][dimensions] order)

## Options
- input_dim, target_dim  : currently set to 44, 1
- unit_type              : 'FC'/'LSTM'/'GRU'/'PILSTM'
- net_width, net_depth   : # of params ~ W<sup>2</sup> D
- batch_size             : minibatch size
- window_size, step_size : for BPTT(h; h') (doi:10.1162/neco.1990.2.4.490)
- learn_init_states      : False/True
- layer_norm             : False/True ([arXiv:1607.06450](https://arxiv.org/abs/1607.06450); modified version of)
- skip_connection        : False/True ([arXiv:1611.01260](https://arxiv.org/abs/1611.01260))
- learn_clock_params     : False/True ([arXiv:1610.09513](https://arxiv.org/abs/1610.09513))
- update_type            : 'sgd'/'momentum'/'nesterov'
- optimizer              : 'vanilla'/'adadelta'/'rmsprop'/'adam'
- frames_per_epoch       : timesteps * batch_size per epoch
- lr_init_val, lr_lower_bound, lr_decay_rate, max_retry : for annealing

# Code reference

## Naming convention 

|  prefix   |      type       |                                   explanation                                    |                          how it's created                           |
|:---------:|:---------------:|:--------------------------------------------------------------------------------:|:-------------------------------------------------------------------:|
|  (none)   | built-in/NumPy  |variable on CPU memory; the usual stuff                                           |the usual way                                                        |
|    v_     |th.SharedVariable|variable on GPU memory; has init value                                            |th.shared                                                            |
|    s_     |  symbolic node  |graph node; has no substance until compiled                                       |tensor literals (tt.alloc, etc)   <br> calculations on v_/s_         |
|port_<i/o>_|  symbolic node  |same as above, but marked for use as       <br> input/output ports for th.function|inputs: tt.scalar, tt.matrix, etc <br> outputs: calculations on v_/s_|
|    f_     | callable object |links built-in/NumPy to input/output ports <br> all v_ updates take place via f_  |th.function                                                          |

* th = theano, tt = theano.tensor
* Prefix not used inside Layer.setup_graph as there is no need for distinction there

<br>


|          suffix           |              explanation              |        
|:-------------------------:|:-------------------------------------:|
|<tensor_name>_[t][b][i/j/k]|[time][batch][hidden] dimensions       |
|    <individual_name>s     |collections (list, dict, etc)          |
|    <any_name>_for_init    |related to options['learn_init_states']|

` [optional], <required>, {default}, (miscellaneous) `

<br>


## Summary of collections in class Net

|     name     |   type    |        element        |                                          explanation                                          |
|:------------:|:---------:|:---------------------:|:---------------------------------------------------------------------------------------------:|
|   _options   |OrderedDict|     str : varies      |training options; saved along with _params                                                     |
|   _params    |OrderedDict|   str : np.ndarray    |learnable parameters (on CPU memory)             <br> used for initialization or as a buffer   |
|  _v_params   |OrderedDict|str : th.SharedVariable|learnable parameters (on GPU memory)                                                           |
|   _v_grads   |    list   |   th.SharedVariable   |loss gradients wrt above; same order as above                                                  |
|_v_prev_states|OrderedDict|str : th.SharedVariable|state values right before this time step         <br> clock and layers own one state each      |
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
- Unroll (https://yjk21.github.io/unrolling.html)
- Simple skip gate (h = h * sig(k) + x * (1. - sig(k)); with k [n_out] initialized negative)
- Reinforcement learning
- Generalize to inequal length sequences (reset signal instead of f_initialize_states)

## Miscellaneous notes
* [SliceableOrderedDict](http://stackoverflow.com/questions/30975339/slicing-a-python-ordereddict)

# Acknowledgements
Consulted code from
- Layer normalization: https://github.com/ryankiros/layer-norm (license not specified)
- Learning rate annealing: https://github.com/KyuyeonHwang/Fractal (Apache-2.0 license)
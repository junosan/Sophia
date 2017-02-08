
# Sophia

*Sophia* (Greek for "wisdom") is a real-time recurrent neural network (RNN)
agent based on Theano.
It focuses on training and evaluating experimental RNN architectures for
regression tasks on long and noisy inputs.
Once trained, the RNN can be used as a plugin to an existing project
(written in any language) to perform real-time estimation by communicating
input/output data via interprocess communication (IPC) or over
a network (TCP).


## Features
- [BPTT(h; h')](https://doi.org/10.1162/neco.1990.2.4.490)
  for more efficient training
- On-line, minibatch training on sequences of unequal lengths
- Inference on ensemble of heterogeneous models
- Scheduled learning rate annealing with patience
- LSTM/GRU with various unit options
 ([weight normalization](https://arxiv.org/abs/1602.07868),
  [layer normalization](https://arxiv.org/abs/1607.06450),
  [batch normalization](https://arxiv.org/abs/1603.09025),
  [residual gate](https://arxiv.org/abs/1611.01260),
  [initial state learning](https://www.cs.toronto.edu/~hinton/csc2535/notes/lec10new.pdf),
  etc.), loss functions
 (L2, L1, [Huber](https://en.wikipedia.org/wiki/Huber_loss)),
  initializations (uniform,
  [orthogonal](https://arxiv.org/abs/1312.6120),
  [Glorot](http://www.jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)),
  and optimizers
 ([Nesterov](https://arxiv.org/abs/1212.0901),
  [Adadelta](https://arxiv.org/abs/1212.5701),
  [RMSProp](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf),
  [Adam](https://arxiv.org/abs/1412.6980))
- Options for learning time-dimension
  (e.g., [Phased LSTM](https://arxiv.org/abs/1610.09513))
  or batch-dimension
  (e.g., [sequence ID](https://doi.org/10.1109/ICASSP.2013.6639211))
  parameters in addition to the standard hidden-dimension parameters
- Option for faster training by unrolling Theano scans (increases compile time)
- Self-documenting object oriented code


## Requirements
Project was developed on Ubuntu 16.04.1 LTS.
Version numbers are not hard requirements.

- Python 2.7 or 3.x (with six 1.10.0)
- Theano (developed on 0.9.0.dev4-py2.7), along with its dependencies
- ZeroMQ (developed on pyzmq 16.0.2)
  : used for communicating real-time input/output data with an external process
- Input and target data preprocessed and saved as separate binary files
  (see below for format)


## Related repository
- [Sibyl](https://github.com/junosan/Sibyl)
  : `sophia.py` may be used as the real-time RNN engine for this project


# How to use

## Selecting the branch
- `master`: for anything other than batch normalization
- `bn`: if using batch normalization (removes certain incompatible features)


## Preparing data
- There is no application specific data processing in this project;
  the code assumes that the input has already been preprocessed, and that any
  post-processing of the output will be done by the receiving side 
- A dataset is contained in a directory with
  - `train.list`: list of training set sequences
  - `dev.list`: list of development set sequences
  - For each sequence, a pair of `sequence_ID.input`/`sequence_ID.target` in
    arbitrary subdirectories (in whichever way that makes sense for the data)
- `.list` files are `'\n'`-delimited plain text lists of relative paths of
  `.input`/`.target` files minus the extensions
  (i.e., one entry of `relative/path/to/sequence_ID` for each sequence)
- Multiple sequences may share the same `sequence_ID` in different
  subdirectores if that makes sense for the data;
  `sequence_ID` can be used as an input feature in this case
- `.input`/`.target` files are binary files each containing a flat array of
  little endian 4-byte floats in [`time`][`dimension`] order
  (stride of `1` in `dimension` axis, stride of `dimension` in `time` axis);
- Time lengths of `.input` & `.target` must match for the same sequence
- Time lengths for different sequences do not need to match, unless using
  batch normalization (where all sequences in a minibatch must be synchronized)


## Training
- Options are configured in `train.py`; see table below
- Instructions for launching a training instance is provided in `train.py`
  heading; multiple instances may be launched if needed
- By providing a `--load_from` flag, the RNN can be trained starting from
  an already trained RNN; this may help getting out of saddle points on
  some tasks

|     option name      |               explanation               |
|:--------------------:|:---------------------------------------:|
| input_dim/target_dim |integers                                 |
|      unit_type       |'fc'/'lstm'/'gru'                        |
|      loss_type       |'l2'/'l1'/'huber'                        |
| net_width/net_depth  |# of params ~ W<sup>2</sup> D            |
|      batch_size      |minibatch size                           |
|window_size/step_size |BPTT(window_size; step_size)             |
|        *_norm        |False/True (weight/layer/batch norm)     |
|    residual_gate     |False/True                               |
|  learn_init_states   |False/True                               |
|  learn_id_embedding  |False/True                               |
|  learn_clock_params  |False/True                               |
|     update_type      |'sgd'/'momentum'/'nesterov'              |
|      force_type      |'vanilla'/'adadelta'/'rmsprop'/'adam'    |
|   frames_per_epoch   |time_indices * batch_size per epoch      |
|   lr_*, max_retry    |for learning rate annealing with patience|
|     unroll_scan      |trades memory consumption & slower compile time for faster training|


## Inference
- `sophia.py` provides an interface for real-time communication with an
  external process which is not necessarily written in Python;
  see `sophia.py` for the communication protocol (handshake + data exchange)
- It is assumed that the same pre-/post-processing is applied to the
  input/output of the RNN as was used during training
- If the use case is simple, it may be directly implemented in Python in
  a manner similar to `sophia.py`


# Notes

## Naming convention
* np = numpy, th = theano, tt = theano.tensor

|  prefix   |      type       |                                   explanation                                    |how it's created|
|:---------:|:---------------:|:--------------------------------------------------------------------------------:|:--------------:|
|  (none)   | built-in/NumPy  |variable on CPU memory                                                            |the usual way   |
|    v_     |th.SharedVariable|variable on GPU memory; has init value                                            |th.shared       |
|    s_     |  symbolic node  |graph node; has no substance until compiled                                       |tensor literals (tt.alloc/zeros/ones, etc) <br> calculations on v_/s_|
|    p_     |  symbolic node  |same as above, but marked for use as       <br> input/output ports for th.function|inputs: tt.scalar/vector/matrix/tensor3, etc <br> outputs: calculations on v_/s_|
|    f_     | callable object |links built-in/NumPy to input/output ports <br> all v_ updates take place via f_  |th.function     |

* Prefix not used inside Layer.setup_graph as there is no need for distinction there

|           suffix          |          explanation           |
|:-------------------------:|:------------------------------:|
|<tensor_name>_[t][b][i/j/k]|[time][batch][hidden] dimensions|
|    <individual_name>s     |collections (list, dict, etc)   |

* `[optional], <required>, {default}, (miscellaneous)`


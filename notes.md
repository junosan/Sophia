* Multi GPU training/inference (synchronous data parallelism) (currently doesn't work due to Theano issues)

# Code reference

## Naming convention 

|  prefix   |      type       |                                   explanation                                    |how it's created|
|:---------:|:---------------:|:--------------------------------------------------------------------------------:|:--------------:|
|  (none)   | built-in/NumPy  |variable on CPU memory                                                            |the usual way   |
|    v_     |th.SharedVariable|variable on GPU memory; has init value                                            |th.shared       |
|    s_     |  symbolic node  |graph node; has no substance until compiled                                       |tensor literals (tt.alloc/zeros/ones, etc) <br> calculations on v_/s_|
|    p_     |  symbolic node  |same as above, but marked for use as       <br> input/output ports for th.function|inputs: tt.scalar/vector/matrix/tensor3, etc <br> outputs: calculations on v_/s_|
|    f_     | callable object |links built-in/NumPy to input/output ports <br> all v_ updates take place via f_  |th.function     |

* `[optional], <required>, {default}, (miscellaneous)`
* np = numpy, th = theano, tt = theano.tensor
* Prefix not used inside Layer.setup_graph as there is no need for distinction there

<br>


|           suffix          |          explanation           |
|:-------------------------:|:------------------------------:|
|<tensor_name>_[t][b][i/j/k]|[time][batch][hidden] dimensions|
|    <individual_name>s     |collections (list, dict, etc)   |

<br>


## Collections in class Net

|    name     |   type    |        element        |                  explanation                  |
|:-----------:|:---------:|:---------------------:|:---------------------------------------------:|
|   options   |OrderedDict|     str : varies      |training options; saved along with params      |
|   params    |OrderedDict|   str : np.ndarray    |learnable parameters (on CPU memory) <br> used only for initialization|
|  prev_dims  |OrderedDict|       str : int       |state variable dimension for each layer        |
|   v_grads   |    list   |   th.SharedVariable   |loss gradients wrt params; same order as params|
|  *_updates  |    list   |       (v_$, s_$)      |give this to th.function to make a function that <br> actually performs the updates when called|
|   slices    |    list   |         Slice         |holds parameters and states for each GPU       | 

<br>

## Collections in class Slice

|    name     |   type    |        element        |                  explanation                  |
|:-----------:|:---------:|:---------------------:|:---------------------------------------------:|
|  v_params   |OrderedDict|str : th.SharedVariable|learnable parameters (on each GPU)             |
|v_prev_states|OrderedDict|str : th.SharedVariable|state values right before this time step starts|

<br>


## Functions in class Net

|            name              |       explanation; input -> output        |
|:----------------------------:|:-----------------------------------------:|
|         _init_params         |add layers, fill params (either freshly, or from file) & prev_dims|
|    _init_shared_variables    |fill v_params[, v_grads], v_prev_states    |
|     _setup_forward_graph     |s_input, s_time, s_id_idx, s_next_prev_idx, v_params, v_prev_states -> s_output, prev_state_updates|
|    _setup_inference_graph    |setup full graph and store ports ([input, time, id_idx], [output]) and updates (prev_state_updates)|
|      _setup_loss_graph       |s_output, s_target, s_step_size -> s_loss  |
|      _setup_grads_graph      |s_loss, v_wrt -> s_grads                   |
|    _setup_optimizer_graph    |s_lr, v_grads -> optim_inits, optim_updates, s_increments|
|    _setup_training_graph     |setup full graph and store ports ([input, target, time, id_idx, step_size; lr], [loss]) and updates (prev_state_updates, grad_updates, optim_inits, optim_param_updates)|
|   compile_f_fwd_propagate    |(training): _ -> th.function([input, target, time, id_idx, step_size], [loss], prev_state_updates) <br> (inference): _ -> th.function([input, time, id_idx], [output], prev_state_updates)|
| compile_f_fwd_bwd_propagate  |_ -> th.function([input, target, time, id_idx, step_size], [loss], grad_updates + prev_state_updates)|
|  compile_f_update_v_params   |_ -> th.function([lr], [], optim_param_updates)|
|compile_f_initialize_optimizer|_ -> th.function([], [], optim_state_inits)|

* *target* means ground truth from files, *output* means what comes out of
  the net
* All updates to shared variables take place via repeated calls to f_*
  (callable objects returned by th.function)

<br>


## TODO
- Reinforcement learning on pre-trained nets

## Miscellaneous notes
* [NumPy/Theano broadcasting rules](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
* [Python 2 & 3 compatibility cheat sheet](http://python-future.org/compatible_idioms.html)
* [SliceableOrderedDict](http://stackoverflow.com/questions/30975339/slicing-a-python-ordereddict)

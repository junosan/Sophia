#==========================================================================#
# Copyright (C) 2016 Hosang Yoon (hosangy@gmail.com) - All Rights Reserved #
# Unauthorized copying of this file, via any medium is strictly prohibited #
#                       Proprietary and confidential                       #
#==========================================================================#

"""
Class for setting up graphs and compiling updater functions
See README.md for a summary of this class
"""

import cPickle as pk
from collections import OrderedDict
import os

from layers import FCLayer, LSTMLayer, GRULayer
from utils import SE_loss, clip_norm, get_random_string
from optimizers import sgd_update, momentum_update, nesterov_update, \
                       vanilla_force, adadelta_force, rmsprop_force, adam_force

import numpy as np
import theano as th
import theano.tensor as tt

class Net():
    def __init__(self, options = None, save_to = None, load_from = None):
        """
        Mode is determined by whether save_to is None or not
        Training:
            <options>   OrderedDict { 'option_name' : option_val }
            <save_to>   str         'path'
            [load_from] str         'path' (set for re-annealing)
        Inference:
            <options>   OrderedDict { 'option_name' : option_val }
            (save_to)   NoneType    (leave as none)
            <load_from> str         'path'
        NOTE: For inference, options['step_size'] and options['batch_size']
              must be specified
        """
        self._configure(options, save_to, load_from)    
        self._init_params(load_from)
        self._init_shared_variables()
        if self._is_training:
            self._setup_training_graph()
        else:
            self._setup_inference_graph()
    
    def _configure(self, options, save_to, load_from):
        if save_to is not None:
            self._is_training = True

            assert options is not None
            self._options = options

            self._save_to = save_to
            self._pfx = ''
            
            if load_from is not None:
                with open(load_from + '/options.pkl', 'rb') as f:
                    loaded_options = pk.load(f)
                    if self._options != loaded_options:
                        print '[Error] Mismatching options in loaded model'
                        assert False
            
            with open(save_to + '/options.pkl', 'wb') as f:
                pk.dump(self._options, f)
        else:
            self._is_training = False
            
            # avoid name clash in ensemble
            self._pfx = get_random_string() + '_'
            
            assert load_from is not None
            with open(load_from + '/options.pkl', 'rb') as f:
                self._options = pk.load(f)
            
            assert 'step_size' in options and 'batch_size' in options
            # to set next_prev_idx = window_size - 1
            self._options['window_size'] = options['step_size']
            self._options['step_size']   = options['step_size']
            self._options['batch_size']  = options['batch_size']
 
    def _init_params(self, load_from):
        """
        Instantiate layers and store their learnable parameters
        and state variables
        Load parameter values from file if applicable
        """
        self._params = OrderedDict()
        self._prev_states = OrderedDict() # not learnable

        def add_states(layer, state_dim):
            if state_dim <= 0:
                return
            self._prev_states[layer.pfx('prev')] = np.zeros \
                ((self._options['batch_size'], state_dim)).astype('float32')
            if self._options['learn_init_states']:
                self._params[layer.pfx('init')] = np.zeros(state_dim) \
                                                    .astype('float32')

        if not self._options['learn_id_embedding']:
            add = 0
        else:
            self._id_embedder = FCLayer(self._pfx + 'FC_id_embedder')
            state_dim = self._id_embedder.add_param \
                (params  = self._params,
                 n_in    = self._options['id_count'],
                 n_out   = self._options['id_embedding_dim'],
                 options = self._options,
                 act     = 'lambda x: x')
            add_states(self._id_embedder, state_dim)
            add = self._options['id_embedding_dim']

        self._layers = []
        D = self._options['net_depth']
        assert D > 0
        for i in range(D):
            self._layers.append(eval(self._options['unit_type'] + 'Layer') \
                    (self._pfx + self._options['unit_type'] + '_' + str(i)))
            state_dim = self._layers[i].add_param \
                (params  = self._params,
                 n_in    = add + (self._options['net_width'] if i > 0 else \
                                  self._options['input_dim']),
                 n_out   = self._options['net_width'],
                 options = self._options)
            add_states(self._layers[i], state_dim)
        
        self._layers.append(FCLayer(self._pfx + 'FC_output'))
        state_dim = self._layers[D].add_param \
                (params  = self._params,
                 n_in    = add + self._options['net_width'],
                 n_out   = self._options['target_dim'],
                 options = self._options,
                 act     = 'lambda x: x')
        add_states(self._layers[D], state_dim)
        
        if load_from is not None:
            len_pfx = len(self._pfx)
            params = np.load(load_from + '/params.npz') # NpzFile object
            for k in self._params.iterkeys():
                self._params[k] = params[k[len_pfx:]] # no pfx in saved params

    def _init_shared_variables(self):
        """
        Initialize shared variables from np.ndarray objects for parameters,
        prev_states, and gradients
        """
        self._v_params = OrderedDict()
        for k, v in self._params.iteritems():
            self._v_params[k] = th.shared(v, name = k)

        self._v_prev_states = OrderedDict()
        for k, v in self._prev_states.iteritems():
            self._v_prev_states[k] = th.shared(v, name = k)

        if self._is_training:
            self._v_grads = \
                [th.shared(v * 0., name = k + '_grad') \
                 for k, v in self._params.iteritems()]

    def _setup_forward_graph(self, s_input_tbi, s_time_tb, s_id_idx_tb,
                                   s_next_prev_idx):
        """
        Specify layer connections
        Layers return their internal states for next time step as
            prev_state_update = (v_prev_state, state[s_next_prev_idx])
        which are collected as a list and returned along with the last
        layer's output
        """
        def get_v_prev_state(layer):
            if layer.pfx('prev') in self._v_prev_states:
                return self._v_prev_states[layer.pfx('prev')]
            else:
                return None
        
        def get_v_init_state(layer):
            if layer.pfx('prev') in self._v_prev_states \
                    and self._options['learn_init_states']:
                return self._v_params[layer.pfx('init')]
            else:
                return None

        def to_one_hot(x_tb, n_class):
            # 2-dim -> 3-dim version of tt.extra_ops.to_one_hot
            x = x_tb.flatten()
            z = tt.zeros((x.shape[0], n_class), dtype = 'float32')
            y = tt.set_subtensor(z[tt.arange(x.shape[0]), x], 1.)
            return tt.reshape(y, (x_tb.shape[0], x_tb.shape[1], n_class))

        if not self._options['learn_id_embedding']:
            cat = lambda s_below_tbj: s_below_tbj
        else:
            s_id_emb_tbi, _ = self._id_embedder.setup_graph \
                (s_below_tbj     = to_one_hot \
                                      (s_id_idx_tb, self._options['id_count']),
                 s_time_tb       = s_time_tb,
                 s_next_prev_idx = s_next_prev_idx,
                 v_params        = self._v_params,
                 v_prev_state_bk = get_v_prev_state(self._id_embedder),
                 v_init_state_k  = get_v_init_state(self._id_embedder))
            cat = lambda s_below_tbj: tt.concatenate([s_below_tbj,
                                                      s_id_emb_tbi], axis = 2)

        D = self._options['net_depth']
        s_outputs = [None] * (D + 1)  # (RNN) * D + FC
        s_outputs.append(s_input_tbi) # put input at index -1
        prev_state_updates = []

        # vertical stack: input -> layer[0] -> ... -> layer[D - 1] -> output
        for i in range(D + 1):
            s_outputs[i], update = self._layers[i].setup_graph \
                (s_below_tbj     = cat(s_outputs[i - 1]),
                 s_time_tb       = s_time_tb,
                 s_next_prev_idx = s_next_prev_idx,
                 v_params        = self._v_params,
                 v_prev_state_bk = get_v_prev_state(self._layers[i]),
                 v_init_state_k  = get_v_init_state(self._layers[i]))
            if update is not None:
                prev_state_updates.append(update)
        
        return s_outputs[D], prev_state_updates

    def _setup_inference_graph(self):
        """
        Connect graphs together for inference and store in/out ports & updates
            Input  : _port_i_input, _port_i_time, _port_i_id_idx
            Output : _port_o_output
            Updates: _prev_state_updates
        """
        self._port_i_input_tbi = tt.ftensor3(name  = 'port_i_input')
        self._port_i_time_tb   = tt.fmatrix (name  = 'port_i_time')
        self._port_i_id_idx_tb = tt.imatrix (name  = 'port_i_id_idx')
        
        # step_size is a compile time constant for inference
        s_next_prev_idx = tt.alloc(np.int32(self._options['step_size'] - 1))

        self._port_o_output_tbi, self._prev_state_updates = \
            self._setup_forward_graph(s_input_tbi     = self._port_i_input_tbi,
                                      s_time_tb       = self._port_i_time_tb,
                                      s_id_idx_tb     = self._port_i_id_idx_tb,
                                      s_next_prev_idx = s_next_prev_idx)

    def _setup_loss_graph(self, s_output_tbi, s_target_tbi, s_step_size):
        """
        Connect a loss function to the graph
        See data.py for explanation of the slicing part
        """
        return SE_loss(s_output_tbi[-s_step_size :],
                       s_target_tbi[-s_step_size :])

    def _setup_grad_updates_graph(self, s_loss, v_wrt, v_grads):
        """
        Connect loss to new values of gradients
            grad_update = (v_grad, s_new_grad)
        NOTE: Takes inputs as lists instead of OrderedDict
        """
        assert len(v_wrt) == len(v_grads)
        s_new_grads = tt.grad(s_loss, wrt = v_wrt)
        if 'grad_norm_clip' in self._options:
            s_new_grads = [clip_norm(s_grad, self._options['grad_norm_clip']) \
                           for s_grad in s_new_grads]
        return zip(v_grads, s_new_grads)

    def _setup_param_updates_graph(self, s_lr, v_params, v_grads):
        """
        Connect learning rate, parameters, gradients, and internal states of
        optimizer to new values of parameters, and inform how optimizer
        should be initiated/updated
            optim_state_init = (v_optim_state, s_init_optim_state)
            param_update     = (v_param, s_new_param) plus 
                               (v_optim_state, s_new_optim_state)
        Assumes that v_grads has been updated prior to applying param_updates
        NOTE: Takes inputs as lists instead of OrderedDict
        """
        assert len(v_params) == len(v_grads)
        optim_f_inits, optim_f_updates, s_forces = \
            eval(self._options['force_type'] + '_force') \
                                                    (options = self._options,
                                                     s_lr    = s_lr,
                                                     v_grads = v_grads)
        assert len(v_params) == len(s_forces)
        optim_u_inits, optim_u_param_updates = \
            eval(self._options['update_type'] + '_update') \
                                                    (options  = self._options,
                                                     v_params = v_params,
                                                     s_forces = s_forces)
        return optim_f_inits + optim_u_inits, \
               optim_f_updates + optim_u_param_updates

    def _setup_training_graph(self):
        """
        Connect graphs together for training and store in/out ports & updates
            Input  : _port_i_input, _port_i_target, _port_i_time,
                     _port_i_id_idx_tb, _port_i_step_size, _port_i_lr, 
            Output : _port_o_loss
            Updates: _prev_state_updates, _grad_updates,
                     _optim_state_inits, _param_updates
        """
        self._port_i_input_tbi  = tt.ftensor3(name = 'port_i_input')
        self._port_i_target_tbi = tt.ftensor3(name = 'port_i_target')
        self._port_i_time_tb    = tt.fmatrix (name = 'port_i_time')
        self._port_i_id_idx_tb  = tt.imatrix (name = 'port_i_id_idx')
        self._port_i_step_size  = tt.iscalar (name = 'port_i_step_size')
        self._port_i_lr         = tt.fscalar (name = 'port_i_lr')

        s_output_tbi, self._prev_state_updates = \
            self._setup_forward_graph \
                (s_input_tbi     = self._port_i_input_tbi,
                 s_time_tb       = self._port_i_time_tb,
                 s_id_idx_tb     = self._port_i_id_idx_tb,
                 s_next_prev_idx = self._port_i_step_size - 1)

        self._port_o_loss = self._setup_loss_graph \
                                (s_output_tbi = s_output_tbi,
                                 s_target_tbi = self._port_i_target_tbi,
                                 s_step_size  = self._port_i_step_size)

        self._grad_updates  = self._setup_grad_updates_graph \
                                (s_loss   = self._port_o_loss,
                                 v_wrt    = self._v_params.values(),
                                 v_grads  = self._v_grads)
        
        self._optim_state_inits, self._param_updates = \
            self._setup_param_updates_graph \
                                (s_lr     = self._port_i_lr,
                                 v_params = self._v_params.values(),
                                 v_grads  = self._v_grads)

    def compile_f_fwd_propagate(self):
        """
        Compile a callable object of signature
            (training)  f(input_tbi, target_tbi, time_tb, id_idx_tb) -> loss
            (inference) f(input_tbi, time_tb, id_idx_tb) -> output_tbi
        As a side effect, calling it updates
            _v_prev_states <- _prev_state_updates
        
        NOTE: Output is a list of np.ndarray (i.e., 0-th element is np.ndarray)
        """
        if self._is_training:
            return th.function \
                (inputs  = [self._port_i_input_tbi, self._port_i_target_tbi,
                            self._port_i_time_tb  , self._port_i_id_idx_tb,
                            self._port_i_step_size],
                 outputs = [self._port_o_loss],
                 updates = self._prev_state_updates,
         on_unused_input = 'raise' if self._options['learn_clock_params'] \
                                   or self._options['learn_id_embedding'] \
                                   else 'ignore')
        else:
            return th.function \
                (inputs  = [self._port_i_input_tbi, self._port_i_time_tb,
                            self._port_i_id_idx_tb],
                 outputs = [self._port_o_output_tbi],
                 updates = self._prev_state_updates,
         on_unused_input = 'raise' if self._options['learn_clock_params'] \
                                   or self._options['learn_id_embedding'] \
                                   else 'ignore')

    def compile_f_fwd_bwd_propagate(self):
        """
        Compile a callable object of signature
            f(input_tbi, target_tbi, time_tb, id_idx_tb) -> loss
        As a side effect, calling it updates
            _v_grads       <- _grad_updates
            _v_prev_states <- _prev_state_updates
        
        NOTE: Output is a list of np.ndarray (i.e., 0-th element is np.ndarray)
              To get loss but not update params (for validation),
              call f_fwd_propagate instead
        """
        assert self._is_training
        return th.function \
            (inputs  = [self._port_i_input_tbi, self._port_i_target_tbi,
                        self._port_i_time_tb  , self._port_i_id_idx_tb,
                        self._port_i_step_size],
             outputs = [self._port_o_loss],
             updates = self._grad_updates + self._prev_state_updates,
     on_unused_input = 'raise' if self._options['learn_clock_params'] \
                               or self._options['learn_id_embedding'] \
                               else 'ignore')
    
    def compile_f_update_v_params(self):
        """
        Compile a callable object of signature
            f(lr) -> None
        As a side effect, calling it updates
            _v_params <- _param_updates
        
        NOTE: f_fwd_bwd_propagate must be called before f_update_v_params
              because it uses gradients stored in _v_grads
              To get loss but not update params (for validation),
              don't call f_update_v_params
        """
        assert self._is_training
        return th.function(inputs  = [self._port_i_lr],
                           outputs = [],
                           updates = self._param_updates)

    def compile_f_initialize_optimizer(self):
        """
        Compile a callable object of signature
            f() -> None
        As a side effect, calling it updates
            v_optim_states <- s_init_optim_states
        Call f_initialize_optimizer when learning rate has changed
        """
        assert self._is_training
        return th.function(inputs  = [],
                           outputs = [],
                           updates = self._optim_state_inits)

    def save_v_params_to_workspace(self, name = 'params'):
        """
        Transfer parameters from GPU to file
        Extension '.npz' appended inside
        """
        assert self._is_training
        for k, v_param in self._v_params.iteritems():
            self._params[k] = v_param.get_value() # pull parameters from GPU

        # There is also savez_compressed, but parameter data
        # doesn't offer much opportunities for compression
        if name is None:
            name = 'params'
        np.savez(self._save_to + '/' + name + '.npz', **self._params)

    def load_v_params_from_workspace(self, name = 'params'):
        """
        Transfer parameters from file to GPU
        Extension '.npz' appended inside
        """
        assert self._is_training
        if name is None:
            name = 'params'
        params = np.load(self._save_to + '/' + name + '.npz') # NpzFile object
        for k, v_param in self._v_params.iteritems():
            v_param.set_value(params[k]) # push parameters to GPU
    
    def remove_params_file_from_workspace(self, name = 'params'):
        """
        Remove temporary file from the workspace
        Extension '.npz' appended inside
        """
        assert self._is_training
        if name is None:
            name = 'params'
        os.remove(self._save_to + '/' + name + '.npz')
    
    def dimensions(self):
        """
        Intended for use during inference
        """
        return self._options['input_dim'], self._options['target_dim']
    
    def save_param(self, param_name, filename):
        """
        Save an individual parameter to file (intended for debugging)
        Cannot be used during training as _params is not updated unless
        save_v_params_to_workspace is called during training
        """
        assert not self._is_training and \
               self._pfx + param_name in self._params
        np.savez(filename,
                 **{ param_name : self._params[self._pfx + param_name] })

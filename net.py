#   Copyright 2017 Hosang Yoon
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""
Class for setting up graphs and compiling updater functions
"""

from __future__ import absolute_import, division, print_function
from six import iterkeys, itervalues, iteritems

import cPickle as pk
from collections import OrderedDict
import os

from layers import FCLayer, LSTMLayer, GRULayer
from utils import l2_loss, l1_loss, huber_loss, clip_norm, get_random_string
from optimizers import sgd_update, momentum_update, nesterov_update, \
                       vanilla_force, adadelta_force, rmsprop_force, adam_force

import numpy as np
import theano as th
import theano.tensor as tt

class Slice():
    def __init__(self, start, stop, context_name = None):
        """
        Batch-dimension slice of the full net, with parameters and states of
        each slice residing in a separate device (GPU) mapped by the given
        context_name
            start, stop     int or NoneType [start, stop) batch index
            [context_name]  str             use str as target for th.shared
                            NoneType        use device set in THEANO_FLAGS
        - [None, None) is equivalent to [:]
        - When there are multiple slices, slice list must be created with
          [start, stop) indices back-to-back in increasing order
        - Parameters of all slices are synchronized every update
        - Make th.SharedVariable's as th.shared(usual_stuff, **s.device)
        """
        assert start is None or stop is None or stop > start
        self._rng   = slice(start, stop)
        self.device = { 'target' : context_name } \
                      if context_name is not None else {}
        self.v_params      = OrderedDict() # { 'str' : th.SharedVariable }
        self.v_prev_states = OrderedDict() # { 'str' : th.SharedVariable }
    
    def transfer(self, s_in): # returns node transferred to this slice's device
        return s_in.transfer(self.device['target']) \
               if self.device != {} else s_in
        # return s_in

    def apply(self, s_full): # slice given node in batch (1-th) dimension
        return self.transfer(s_full[:, self._rng])
    
    def get_size(self, full_size):
        return ((full_size if self._rng.stop  is None else self._rng.stop )
              - (0         if self._rng.start is None else self._rng.start))


class Net():
    def __init__(self, options,
                       save_to = None, load_from = None, c_names = None):
        """
        Mode is determined by whether save_to is None or not

        (common)
            <options>   OrderedDict { 'option_name' : option_val }
            [c_names]   list of str [context names for each GPU]
                                    (multi GPU mode ; THEANO_FLAGS=contexts=$)
                        NoneType    (single GPU mode; THEANO_FLAGS=device=$)
        (training)
            <save_to>   str         'workspace_dir'
            [load_from] str         'workspace_dir' (if re-annealing)
                        NoneType    (if training fresh)
        (inference)
            (save_to)   NoneType    (leave as none)
            <load_from> str         'workspace_dir'
        
        NOTE: For inference, options['step_size'] and options['batch_size']
              must be specified
        """
        self._configure(options, save_to, load_from, c_names)
        self._init_params(load_from)
        self._init_shared_variables()
        if self._is_training:
            self._setup_training_graph()
        else:
            self._setup_inference_graph()
    
    def _configure(self, options, save_to, load_from, c_names):
        if save_to is not None:
            self._is_training = True

            self._options = options
            self._save_to = save_to
            self._pfx = ''
            
            if load_from is not None:
                with open(load_from + '/options.pkl', 'rb') as f:
                    loaded_options = pk.load(f)
                    assert self._options == loaded_options, \
                           'Mismatching options in loaded model'
            
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
        
        if c_names is not None:
            n = self._options['batch_size']
            m = len(c_names)

            # distribute batches in n // m chunks
            # if n % m > 0, it is added to the last device
            i = [(n // m) * k for k in range(m)] + [n]
            self._slices = [Slice(*args) for args in zip(i, i[1 :], c_names)]
        else:
            self._slices = [Slice(None, None)]
        
        # this is where gradients and optimizer states are stored
        self._device = self._slices[0].device
 
    def _init_params(self, load_from):
        """
        Instantiate layers and store their learnable parameters and state info
        Load parameter values from file if applicable
        """
        self._params = OrderedDict()
        self._prev_dims = OrderedDict() # not learnable

        def add_states(layer, state_dim):
            if state_dim <= 0:
                return
            self._prev_dims[layer.pfx('prev')] = state_dim
            if self._options['learn_init_states']:
                self._params[layer.pfx('init')] = np.zeros(state_dim) \
                                                    .astype('float32')


        # optional ID embedder
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

        # main recurrent layers
        unit = self._options['unit_type'].upper()
        self._layers = []
        D = self._options['net_depth']
        assert D > 0
        for i in range(D):
            self._layers.append(eval(unit + 'Layer') \
                                    (self._pfx + unit + '_' + str(i)))
            state_dim = self._layers[i].add_param \
                (params  = self._params,
                 n_in    = add + (self._options['net_width'] if i > 0 else \
                                  self._options['input_dim']),
                 n_out   = self._options['net_width'],
                 options = self._options)
            add_states(self._layers[i], state_dim)
        
        # final FCLayer for dimension compression
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
            for k in iterkeys(self._params):
                self._params[k] = params[k[len_pfx :]] # no pfx in saved params

    def _init_shared_variables(self):
        """
        Initialize shared variables from np.ndarray objects for parameters,
        prev_states, and gradients
        """
        for s in self._slices:
            dev = s.device['target'] + '_' if s.device != {} else ''

            for k, v in iteritems(self._params):
                s.v_params[k] = th.shared(v, name = dev + k, **s.device)

            for k, d in iteritems(self._prev_dims):
                v = np.zeros((s.get_size(self._options['batch_size']), d)) \
                      .astype('float32')
                s.v_prev_states[k] = th.shared(v, name = dev + k, **s.device)

        if self._is_training:
            self._v_grads = \
                [th.shared(v * 0., name = k + '_grad', **self._device) \
                 for k, v in iteritems(self._params)]

    def _setup_forward_graph(self, s_input_tbi, s_time_tb, s_id_idx_tb,
                                   s_next_prev_idx, v_params, v_prev_states):
        """
        Specify layer connections
        Layers return their internal states for next time step as
            prev_state_update = (v_prev_state, state[s_next_prev_idx])
        which are collected as a list and returned along with the last
        layer's output
        """
        def get_v_prev_state(layer):
            if layer.pfx('prev') in v_prev_states:
                return v_prev_states[layer.pfx('prev')]
            else:
                return None
        
        def get_v_init_state(layer):
            if layer.pfx('prev') in v_prev_states \
                    and self._options['learn_init_states']:
                return v_params[layer.pfx('init')]
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
                 v_params        = v_params,
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
                 v_params        = v_params,
                 v_prev_state_bk = get_v_prev_state(self._layers[i]),
                 v_init_state_k  = get_v_init_state(self._layers[i]))
            if update is not None:
                prev_state_updates.append(update)
        
        return s_outputs[D], prev_state_updates

    def _setup_inference_graph(self):
        """
        Connect graphs together for inference and store in/out ports & updates
            inputs  : input, time, id_idx
            outputs : output
            updates : prev_states
        """
        p_input_tbi = tt.ftensor3(name  = 'port_i_input')
        p_time_tb   = tt.fmatrix (name  = 'port_i_time')
        p_id_idx_tb = tt.imatrix (name  = 'port_i_id_idx')
        
        # step_size is a compile time constant for inference
        s_next_prev_idx = tt.alloc(np.int32(self._options['step_size'] - 1))

        outputs = []
        self._prev_state_updates = []

        for s in self._slices:
            s_output_tbi, prev_state_updates = self._setup_forward_graph \
                (s_input_tbi     = s.apply(p_input_tbi),
                 s_time_tb       = s.apply(p_time_tb),
                 s_id_idx_tb     = s.apply(p_id_idx_tb),
                 s_next_prev_idx = s.transfer(s_next_prev_idx),
                 v_params        = s.v_params,
                 v_prev_states   = s.v_prev_states)
            outputs += [self.transfer(s_output_tbi)]
            self._prev_state_updates += prev_state_updates

        # merge outputs from all slices
        p_output_tbi = tt.concatenate(outputs, axis = 1)

        self._prop_i_ports   = [p_input_tbi, p_time_tb, p_id_idx_tb]
        self._prop_o_ports   = [p_output_tbi]

    def _setup_loss_graph(self, s_output_tbi, s_target_tbi, s_step_size):
        """
        Connect a loss function to the graph
        See data.py for explanation of the slicing part
        """
        s_sliced_output_tbi = s_output_tbi[-s_step_size :]
        s_sliced_target_tbi = s_target_tbi[-s_step_size :]

        if self._options['loss_type'] == 'l2':
            return l2_loss(s_sliced_output_tbi, s_sliced_target_tbi)
        if self._options['loss_type'] == 'l1':
            return l1_loss(s_sliced_output_tbi, s_sliced_target_tbi)
        if self._options['loss_type'] == 'huber':
            delta = self._options['huber_delta']
            return huber_loss(s_sliced_output_tbi, s_sliced_target_tbi, delta)
        
        assert False, 'Invalid loss_type option'
        return tt.alloc(np.float32(0.))

    def _setup_grads_graph(self, s_loss, v_wrt):
        """
        Connect loss to new values of gradients
        - NOTE: v_wrt must be a list instead of OrderedDict
        """
        assert type(v_wrt) is list
        s_grads = tt.grad(s_loss, wrt = v_wrt)
        if 'grad_norm_clip' in self._options:
            s_grads = [clip_norm(s_grad, self._options['grad_norm_clip']) \
                           for s_grad in s_grads]
        return s_grads # list of nodes

    def _setup_optimizer_graph(self, s_lr, v_grads):
        """
        Connect learning rate, gradients, and internal states of optimizer
        to increments for parameters, and inform how optimizer should be
        initiated/updated
        - Returns lists of
            optim_init   = (v_optim_state, s_init_optim_state)
            optim_update = (v_optim_state, s_new_optim_state)
            s_increment    (to update s.v_param <- s.v_param + s_increment)
        - Assumes that v_grads has been updated prior to applying updates here
        - NOTE: v_grads must be a list instead of OrderedDict
        """
        assert type(v_grads) is list

        # same shapes and orders as v_grads
        ones = [np.ones_like(p).astype('float32') \
                for p in itervalues(self._params)]

        optim_f_inits, optim_f_updates, s_forces = \
            eval(self._options['force_type'] + '_force') \
                                                    (options = self._options,
                                                     ones    = ones,
                                                     s_lr    = s_lr,
                                                     v_grads = v_grads,
                                                     device  = self._device)

        optim_u_inits, optim_u_updates, s_increments = \
            eval(self._options['update_type'] + '_update') \
                                                    (options  = self._options,
                                                     ones     = ones,
                                                     s_forces = s_forces,
                                                     device   = self._device)
        
        return (optim_f_inits + optim_u_inits,
                optim_f_updates + optim_u_updates,
                s_increments)

    def _setup_training_graph(self):
        """
        Connect graphs together for training and store in/out ports & updates
        (propagation)  inputs  : input, target, time, id_idx, step_size
                       outputs : loss
                       updates : prev_states[, grads]
        (param update) inputs  : lr
                       outputs : None
                       updates : params
        (optim init)   inputs  : None
                       outputs : None
                       updates : optimizer states
        """
        p_input_tbi  = tt.ftensor3(name = 'i_port_input')
        p_target_tbi = tt.ftensor3(name = 'i_port_target')
        p_time_tb    = tt.fmatrix (name = 'i_port_time')
        p_id_idx_tb  = tt.imatrix (name = 'i_port_id_idx')
        p_step_size  = tt.iscalar (name = 'i_port_step_size')
        p_lr         = tt.fscalar (name = 'i_port_lr')

        self._prev_state_updates = []
        losses = [] # list of s_loss
        gradss = [] # list of s_grads (i.e., list of list)

        for s in self._slices:
            s_step_size = s.transfer(p_step_size)
            s_output_tbi, prev_state_updates = self._setup_forward_graph \
                (s_input_tbi     = s.apply(p_input_tbi),
                 s_time_tb       = s.apply(p_time_tb),
                 s_id_idx_tb     = s.apply(p_id_idx_tb),
                 s_next_prev_idx = s_step_size - 1,
                 v_params        = s.v_params,
                 v_prev_states   = s.v_prev_states)
            self._prev_state_updates += prev_state_updates

            s_loss = self._setup_loss_graph \
                (s_output_tbi = s_output_tbi,
                 s_target_tbi = s.apply(p_target_tbi),
                 s_step_size  = s_step_size)
            losses += [self.transfer(s_loss)]

            s_grads = self._setup_grads_graph \
                (s_loss = s_loss,
                 v_wrt  = list(itervalues(s.v_params)))
            gradss += [[self.transfer(s_grad) for s_grad in s_grads]]
        
        # sum losses and grads from all slices
        p_loss = sum(losses)
        s_new_grads = [sum(grad_tuple) for grad_tuple in zip(*gradss)]
        self._grad_updates = [u for u in zip(self._v_grads, s_new_grads)]

        self._optim_inits, self._optim_param_updates, s_increments = \
            self._setup_optimizer_graph(s_lr    = self.transfer(p_lr),
                                        v_grads = self._v_grads)

        for s in self._slices:
            self._optim_param_updates += \
                [(p, p + i) for p, i in zip(s.v_params.values(), s_increments)]

        self._prop_i_ports   = [p_input_tbi, p_target_tbi, p_time_tb,
                                p_id_idx_tb, p_step_size]
        self._prop_o_ports   = [p_loss]
        self._update_i_ports = [p_lr]

    def compile_f_fwd_propagate(self):
        """
        Compile a callable object of signature
            (training)  f(input_tbi, target_tbi, time_tb,
                          id_idx_tb, step_size) -> [loss]
            (inference) f(input_tbi, time_tb, id_idx_tb) -> [output_tbi]
        As a side effect, calling it updates
            v_prev_states
        
        - Output is a list of np.ndarray (i.e., 0-th element is np.ndarray)
          whether scalar (loss) or tensor3 (output_tbi)
        """
        on_unused_input = 'raise' if self._options['learn_id_embedding'] \
                                  else 'ignore'
        return th.function(inputs  = self._prop_i_ports,
                           outputs = self._prop_o_ports,
                           updates = self._prev_state_updates,
                           on_unused_input = on_unused_input)

    def compile_f_fwd_bwd_propagate(self):
        """
        Compile a callable object of signature
            f(input_tbi, target_tbi, time_tb, id_idx_tb, step_size) -> [loss]
        As a side effect, calling it updates
            v_grads, v_prev_states
        
        - Output is a list of np.ndarray (i.e., loss = np.asscalar(output[0]))
        - For validation (obtain loss only), call f_fwd_propagate instead
        """
        assert self._is_training
        on_unused_input = 'raise' if self._options['learn_id_embedding'] \
                                  else 'ignore'
        return th.function(inputs  = self._prop_i_ports,
                           outputs = self._prop_o_ports,
                           updates = (self._grad_updates
                                      + self._prev_state_updates),
                           on_unused_input = on_unused_input)
    
    def compile_f_update_v_params(self):
        """
        Compile a callable object of signature
            f(lr) -> None
        As a side effect, calling it updates
            v_optim_states, v_params
        
        - f_fwd_bwd_propagate must be called before f_update_v_params
          because it uses gradients stored in _v_grads
        - For validation, don't call f_update_v_params
        """
        assert self._is_training
        return th.function(inputs  = self._update_i_ports,
                           outputs = [],
                           updates = self._optim_param_updates)

    def compile_f_initialize_optimizer(self):
        """
        Compile a callable object of signature
            f() -> None
        As a side effect, calling it initializes
            v_optim_states
        Call f_initialize_optimizer when learning rate has changed
        """
        assert self._is_training
        return th.function(inputs  = [],
                           outputs = [],
                           updates = self._optim_inits)

    def save_to_workspace(self, name = None):
        """
        Transfer parameters from GPU to file
        """
        assert self._is_training
        sfx = name if name is not None else ''

        # v_params in all slices are in sync, so we just use 0-th
        for k, v_param in iteritems(self._slices[0].v_params):
            self._params[k] = v_param.get_value() # pull from GPU

        # There is also savez_compressed, but parameter data
        # doesn't offer much opportunities for compression
        np.savez(self._save_to + '/params' + sfx + '.npz', **self._params)

    def load_from_workspace(self, name = None):
        """
        Transfer parameters from file to GPU
        """
        assert self._is_training
        sfx = name if name is not None else ''
        
        # ret = NpzFile object
        params = np.load(self._save_to + '/params' + sfx + '.npz')
        for s in self._slices:
            for k, v_param in iteritems(s.v_params):
                v_param.set_value(params[k]) # push to GPU
    
    def remove_from_workspace(self, name = None):
        """
        Remove temporary file from the workspace
        """
        assert self._is_training
        sfx = name if name is not None else ''

        os.remove(self._save_to + '/params' + sfx + '.npz')
    
    def transfer(self, s_in):
        """
        Return given node transferred to Net's device (same as 0-th Slice)
        """
        return s_in.transfer(self._device['target']) \
               if self._device != {} else s_in
        # return s_in

    def dimensions(self):
        return self._options['input_dim'], self._options['target_dim']
    
    def n_weights(self):
        return sum(p.size for p in itervalues(self._params))

    def save_param(self, param_name, file_name):
        """
        Save an individual parameter to file (intended for debugging)
        Cannot be used during training as _params is not updated unless
        save_to_workspace is called during training
        """
        assert not self._is_training and \
               self._pfx + param_name in self._params
        np.savez(file_name,
                 **{ param_name : self._params[self._pfx + param_name] })

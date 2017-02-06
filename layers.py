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
Abstract Layer class and specific layer types as derived classes
"""

from __future__ import absolute_import, division, print_function
from six import with_metaclass

from abc import ABCMeta, abstractmethod
import numpy as np
import theano as th
import theano.tensor as tt
from utils import unif_weight

def cut0(x, n, stride):
    return x[n * stride : (n + 1) * stride]    # assumes x.ndim > 0

def cut1(x, n, stride):
    return x[:, n * stride : (n + 1) * stride] # assumes x.ndim > 1

def weight_norm(W_jk, g_k): # weight normalization
    return g_k * W_jk / W_jk.norm(2, axis = 0, keepdims = True)

class Layer(with_metaclass(ABCMeta)):
    def __init__(self, name):
        self.name = name # NOTE: each layer should be given a unique name

    @abstractmethod
    def add_param(self, params, n_in, n_out, options, **kwargs):
        """
        Add learnable parameters to params and report size of state variables

        Inputs:
           &params  OrderedDict { str : np.ndarray }
            n_in    int
            n_out   int
            options OrderedDict { str : varies     }
            kwargs  dict        { str : varies     } (for any other settings)
        Returns:
            state_dim
        
        - Always use pfx(name) for parameter names
        - By adding learnable parameters as params[pfx(name)], expect
            v_params[pfx(name)]     (same dimensions as params[pfx(name)])
          to be provided in setup_graph
        - Dimensions of v_prev_state_bk and s_init_state_k provided in
          setup_graph depend on state_dim returned here
        """
        pass
    
    @abstractmethod
    def setup_graph(self, s_below_tbj, s_time_tb, s_next_prev_idx,
                    v_params, v_prev_state_bk, v_init_state_k):
        """
        Connect input nodes to outputs and prepare prev_state_update
        for the next time step

        Inputs:
            s_below_tbj     symbolic          [n_steps][batch_size][n_in]
            s_time_tb       symbolic          [n_steps][batch_size]
            s_next_prev_idx symbolic          (int32 index)
            v_params        OrderedDict       { str : th.SharedVariable }
            v_prev_state_bk th.SharedVariable [batch_size][state_dim]
                            NoneType          (if state_dim == 0)
            v_init_state_k  th.SharedVariable [state_dim]
                                              (if options['learn_init_states'])
                            NoneType          (if not above or state_dim == 0)
        Returns:
            tuple of 
                s_output_tbi, prev_state_update
            where prev_state_update =
                (v_prev_state_bk, state[s_next_prev_idx]) (if state_dim > 0)
                None                                      (if state_dim == 0)
        
        - If state_dim > 0, v_prev_state_bk holds the state right before
          the 0-th time index of this time step
        - If state_dim > 0, step function should use tt.switch to pick
              states from (time index - 1)  (if time_b > 0.)
              s_init_state_k                (else)
          as the state values carried from (time index - 1)
        - Receiving side should check if prev_state_update is None or not
        """
        pass
    
    def pfx(self, s):
        return '%s_%s' % (self.name, s)

    def add_clock_params(self, params, n_out, options):
        # clk_t : period ~ exp(Uniform(lo, hi))
        # clk_s : shift  ~ Uniform(0, clk_t)
        params[self.pfx('clk_t')] = np.exp(np.random.uniform \
                                            (low  = options['clock_t_exp_lo'],
                                             high = options['clock_t_exp_hi'],
                                             size = (n_out))).astype('float32')
        params[self.pfx('clk_s')] = (params[self.pfx('clk_t')]
                                     * np.random.uniform(size = (n_out)) \
                                       .astype('float32'))
        self.clock_r_on      = options['clock_r_on']
        self.clock_leak_rate = options['clock_leak_rate']
    
    def setup_clock_graph(self, s_time_tb, t_i, s_i):
        phi_tbi = tt.mod(s_time_tb[:, :, None] - s_i, t_i) / t_i # broadcasts
        r_on  = self.clock_r_on
        alpha = self.clock_leak_rate
        return tt.switch(phi_tbi < r_on / 2., 2. * phi_tbi / r_on,
               tt.switch(phi_tbi < r_on     , 2. - 2. * phi_tbi / r_on,
                                             alpha * phi_tbi))
    

class FCLayer(Layer):
    def add_param(self, params, n_in, n_out, options, **kwargs):
        # Pass, e.g., act = 'lambda x: tt.tanh(x)' as last argument
        assert callable(eval(kwargs['act']))
        self._act = eval(kwargs['act'])

        params[self.pfx('W')] = unif_weight(options, n_in, n_out)
        params[self.pfx('b')] = unif_weight(options, n_out)

        self.use_weight_norm = options['weight_norm']
        if self.use_weight_norm: # scaled to make same norm as unif_weight
            params[self.pfx('wn_Wg')] = (np.euler_gamma * options['init_scale']
                            * np.sqrt(n_in) * np.ones(n_out).astype('float32'))

        self.use_res_gate = options['residual_gate'] and n_in == n_out
        if self.use_res_gate:
            params[self.pfx('rg_k')] = -1. * np.ones(n_out).astype('float32')
        
        return 0 # no state variables

    def setup_graph(self, s_below_tbj, s_time_tb, s_next_prev_idx,
                    v_params, v_prev_state_bk, v_init_state_k):
        v_param = lambda name: v_params[self.pfx(name)]

        W_ji = weight_norm(v_param('W'), v_param('wn_Wg')) \
               if self.use_weight_norm else v_param('W')
        h_tbi = self._act(tt.dot(s_below_tbj, W_ji) + v_param('b'))
        
        if not self.use_res_gate:
            out_tbi = h_tbi
        else:
            g_i = tt.nnet.sigmoid(v_param('rg_k'))
            out_tbi = g_i * h_tbi + (1. - g_i) * s_below_tbj

        return out_tbi, None # no prev_state_update


class LSTMLayer(Layer):
    def add_param(self, params, n_in, n_out, options, **kwargs):
        self.n_steps         = options['window_size']
        self.n_out           = n_out
        self.use_peephole    = options['lstm_peephole']
        self.use_weight_norm = options['weight_norm']
        self.use_clock       = options['learn_clock_params']
        self.use_res_gate    = options['residual_gate'] and n_in == n_out
        self.unroll_scan     = options['unroll_scan']

        # input to (i, f, c, o) [n_in][4 * n_out]
        params[self.pfx('W')] = np.concatenate \
                                    ([unif_weight(options, n_in, n_out) \
                                      for _ in range(4)], axis = 1)
        params[self.pfx('b')] = unif_weight(options, 4 * n_out)
        # hidden to (i, f, c, o) [n_out][4 * n_out]
        params[self.pfx('U')] = np.concatenate \
                                    ([unif_weight(options, n_out, n_out) \
                                      for _ in range(4)], axis = 1)
        
        if self.use_peephole:
            params[self.pfx('p')] = unif_weight(options, 3 * n_out)

        if self.use_weight_norm: # scaled to make same norm as unif_weight
            params[self.pfx('wn_Wg')] = (np.euler_gamma * options['init_scale']
                    * np.sqrt(n_in ) * np.ones(4 * n_out).astype('float32'))
            params[self.pfx('wn_Ug')] = (np.euler_gamma * options['init_scale']
                    * np.sqrt(n_out) * np.ones(4 * n_out).astype('float32'))

        if self.use_clock:
            self.add_clock_params(params, n_out, options)

        if self.use_res_gate:
            params[self.pfx('rg_k')] = -1. * np.ones(n_out).astype('float32')

        return 2 * n_out # h, c

    def setup_graph(self, s_below_tbj, s_time_tb, s_next_prev_idx,
                    v_params, v_prev_state_bk, v_init_state_k):
        v_param = lambda name: v_params[self.pfx(name)]
        n_out   = self.n_out

        W_j4i  = weight_norm(v_param('W'), v_param('wn_Wg')) \
                 if self.use_weight_norm else v_param('W')
        b_4i   = v_param('b')
        x_tb4i = tt.dot(s_below_tbj, W_j4i) + b_4i

        use_init = v_init_state_k is not None
        init_h_i = cut0(v_init_state_k, 0, n_out) if use_init else 0.
        init_c_i = cut0(v_init_state_k, 1, n_out) if use_init else 0.
        U_i4i    = weight_norm(v_param('U'), v_param('wn_Ug')) \
                   if self.use_weight_norm else v_param('U')
        p_3i     = v_param('p') if self.use_peephole else \
                   tt.zeros(3 * n_out).astype('float32')
        non_seqs = [init_h_i, init_c_i, U_i4i, p_3i]
        
        mask_tbi = self.setup_clock_graph \
                       (s_time_tb, v_param('clk_t'), v_param('clk_s')) \
                   if self.use_clock else \
                   tt.ones((self.n_steps, 1, 1), dtype = 'float32')
        
        def step(x_b4i, time_b, mask_bi, prev_h_bi, prev_c_bi, *args):
            prev_h_bi = tt.switch(time_b[:, None] > 0., prev_h_bi, init_h_i)
            prev_c_bi = tt.switch(time_b[:, None] > 0., prev_c_bi, init_c_i)
            
            preact_b4i = x_b4i + tt.dot(prev_h_bi, U_i4i)

            i_bi = tt.nnet.sigmoid(cut1(preact_b4i, 0, n_out)
                                 + cut0(p_3i, 0, n_out) * prev_c_bi)
            f_bi = tt.nnet.sigmoid(cut1(preact_b4i, 1, n_out)
                                 + cut0(p_3i, 1, n_out) * prev_c_bi)

            c_bi = (i_bi * tt.tanh(cut1(preact_b4i, 2, n_out))
                    + f_bi * prev_c_bi)
            if self.use_clock:
                c_bi = mask_bi * c_bi + (1. - mask_bi) * prev_c_bi

            o_bi = tt.nnet.sigmoid(cut1(preact_b4i, 3, n_out)
                                 + cut0(p_3i, 2, n_out) * c_bi)
            
            h_bi = o_bi * tt.tanh(c_bi)
            if self.use_clock:
                h_bi = mask_bi * h_bi + (1. - mask_bi) * prev_h_bi 

            return h_bi, c_bi

        if not self.unroll_scan:
            ((h_tbi, c_tbi), _) = th.scan(step,
                      sequences     = [x_tb4i, s_time_tb, mask_tbi],
                      outputs_info  = [cut1(v_prev_state_bk, 0, n_out),
                                       cut1(v_prev_state_bk, 1, n_out)],
                      non_sequences = non_seqs,
                      n_steps       = self.n_steps,
                      name          = self.pfx('scan'),
                      strict        = True)
        else:
            h_list = [cut1(v_prev_state_bk, 0, n_out)]
            c_list = [cut1(v_prev_state_bk, 1, n_out)]
            for t in range(self.n_steps):
                h_bi, c_bi = step(x_tb4i[t], s_time_tb[t], mask_tbi[t],
                                  h_list[-1], c_list[-1], *non_seqs)
                h_list.append(h_bi)
                c_list.append(c_bi)
            h_tbi = tt.stack(h_list[1 :], axis = 0)
            c_tbi = tt.stack(c_list[1 :], axis = 0)

        if not self.use_res_gate:
            out_tbi = h_tbi
        else:
            g_i = tt.nnet.sigmoid(v_param('rg_k'))
            out_tbi = g_i * h_tbi + (1. - g_i) * s_below_tbj

        return out_tbi, (v_prev_state_bk,
                         tt.concatenate([h_tbi[s_next_prev_idx],
                                         c_tbi[s_next_prev_idx]], axis = 1))


class GRULayer(Layer):
    def add_param(self, params, n_in, n_out, options, **kwargs):
        self.n_steps         = options['window_size']
        self.n_out           = n_out
        self.use_weight_norm = options['weight_norm']
        self.use_clock       = options['learn_clock_params']
        self.use_res_gate    = options['residual_gate'] and n_in == n_out
        self.unroll_scan     = options['unroll_scan']

        # input to (r, u, c) [n_in][3 * n_out]
        params[self.pfx('W')] = np.concatenate \
                                    ([unif_weight(options, n_in, n_out) \
                                      for _ in range(3)], axis = 1)
        params[self.pfx('b')] = unif_weight(options, 3 * n_out)
        # hidden to (r, u, c) [n_out][3 * n_out]
        params[self.pfx('U')] = np.concatenate \
                                    ([unif_weight(options, n_out, n_out) \
                                      for _ in range(3)], axis = 1)
                
        if self.use_weight_norm: # scaled to make same norm as unif_weight
            params[self.pfx('wn_Wg')] = (np.euler_gamma * options['init_scale']
                    * np.sqrt(n_in ) * np.ones(3 * n_out).astype('float32'))
            params[self.pfx('wn_Ug')] = (np.euler_gamma * options['init_scale']
                    * np.sqrt(n_out) * np.ones(3 * n_out).astype('float32'))
        
        if self.use_clock:
            self.add_clock_params(params, n_out, options)

        if self.use_res_gate:
            params[self.pfx('rg_k')] = -1. * np.ones(n_out).astype('float32')
        
        return n_out # h

    def setup_graph(self, s_below_tbj, s_time_tb, s_next_prev_idx,
                    v_params, v_prev_state_bk, v_init_state_k):
        v_param = lambda name: v_params[self.pfx(name)]
        n_out   = self.n_out

        W_j3i  = weight_norm(v_param('W'), v_param('wn_Wg')) \
                 if self.use_weight_norm else v_param('W')
        b_3i   = v_param('b')
        x_tb3i = tt.dot(s_below_tbj, W_j3i) + b_3i
        
        init_h_i = v_init_state_k if v_init_state_k is not None else 0.
        U_i3i    = weight_norm(v_param('U'), v_param('wn_Ug')) \
                   if self.use_weight_norm else v_param('U')
        non_seqs = [init_h_i, U_i3i]

        mask_tbi = self.setup_clock_graph \
                       (s_time_tb, v_param('clk_t'), v_param('clk_s')) \
                   if self.use_clock else \
                   tt.ones((self.n_steps, 1, 1), dtype = 'float32')

        def step(x_b3i, time_b, mask_bi, prev_h_bi, *args):
            prev_h_bi = tt.switch(time_b[:, None] > 0., prev_h_bi, init_h_i)
            
            preact_b2i = (cut1(x_b3i, 0, 2 * n_out)
                        + tt.dot(prev_h_bi, cut1(U_i3i, 0, 2 * n_out)))

            r_bi = tt.nnet.sigmoid(cut1(preact_b2i, 0, n_out))
            u_bi = tt.nnet.sigmoid(cut1(preact_b2i, 1, n_out))

            c_bi = tt.tanh(cut1(x_b3i, 2, 1 * n_out)
                         + r_bi * tt.dot(prev_h_bi, cut1(U_i3i, 2, 1 * n_out)))

            h_bi = (1. - u_bi) * prev_h_bi + u_bi * c_bi
            if self.use_clock:
                h_bi = mask_bi * h_bi + (1. - mask_bi) * prev_h_bi

            return h_bi

        if not self.unroll_scan:
            h_tbi, _ = th.scan(step,
                               sequences     = [x_tb3i, s_time_tb, mask_tbi],
                               outputs_info  = v_prev_state_bk,
                               non_sequences = non_seqs,
                               n_steps       = n_steps,
                               name          = self.pfx('scan'),
                               strict        = True)
        else:
            h_list = [v_prev_state_bk]
            for t in range(self.n_steps):
                h_list.append(step(x_tb3i[t], s_time_tb[t], mask_tbi[t],
                                   h_list[-1], *non_seqs))
            h_tbi = tt.stack(h_list[1 :], axis = 0)

        if not self.use_res_gate:
            out_tbi = h_tbi
        else:
            g_i = tt.nnet.sigmoid(v_param('rg_k'))
            out_tbi = g_i * h_tbi + (1. - g_i) * s_below_tbj

        return out_tbi, (v_prev_state_bk, h_tbi[s_next_prev_idx])

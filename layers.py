#==========================================================================#
# Copyright (C) 2016 Hosang Yoon (hosangy@gmail.com) - All Rights Reserved #
# Unauthorized copying of this file, via any medium is strictly prohibited #
#                       Proprietary and confidential                       #
#==========================================================================#

"""
Abstract Layer class and specific layer types as derived classes
"""

from abc import ABCMeta, abstractmethod
import numpy as np
import theano as th
import theano.tensor as tt
import theano.gpuarray.dnn as cu
from utils import unif_weight

def cut0(x, n, stride):
    return x[n * stride : (n + 1) * stride]    # assumes x.ndim > 0

def cut1(x, n, stride):
    return x[:, n * stride : (n + 1) * stride] # assumes x.ndim > 1

class Layer():
    __metaclass__ = ABCMeta

    def __init__(self, name):
        self.name   = name # NOTE: each layer should be given a unique name
        self.bn_eps = 1e-4 # default value in Theano; cuDNN asserts eps >= 1e-5

    @abstractmethod
    def add_param(self, params, n_in, n_out, options, **kwargs):
        """
        Add learnable parameters to params and report dimensions of state
        variables & batch statistics variables (if applicable)

        Inputs:
           &params  OrderedDict { str : np.ndarray }
            n_in    int
            n_out   int
            options OrderedDict { str : varies     }
            kwargs  dict        { str : varies     } (for any other settings)
        Returns:
            state_dim, stat_dim
        
        - Always use pfx(name) for parameter names
        - By adding learnable parameters as params[pfx(name)], expect
            v_params[pfx(name)]     (same dimensions as params[pfx(name)])
          to be provided in setup_graph
        - Return values determine the last dimension of non-learnable 
          th.SharedVariable's used in setup_graph:
              k = state_dim (> 0 for recurrent layers with states)
              l = stat_dim  (> 0 for layers with batch_norm statistics)
                  0         (if not options['batch_norm'])
        """
        pass
    
    @abstractmethod
    def setup_graph(self, s_below_tbj, s_time_t, s_next_prev_idx,
                    v_params, v_prev_state_bk, v_init_state_k, v_stat_Tsl):
        """
        Connect input nodes to outputs, prepare prev_state_update for the
        next time step, and report batch statistics (if applicable)

        Inputs:
            s_below_tbj     symbolic          [n_steps][batch_size][n_in]
            s_time_t        symbolic          [n_steps] (int32 index)
            s_next_prev_idx symbolic          scalar    (int32 index)
            v_params        OrderedDict       { str : th.SharedVariable }
            v_prev_state_bk th.SharedVariable [batch_size][state_dim]
                            tt.alloc(0.)      (if state_dim == 0)
            v_init_state_k  th.SharedVariable [state_dim]
                                              (if options['learn_init_states'])
                            tt.alloc(0.)      (if not above or state_dim == 0)
            v_stat_Tsl      th.SharedVariable [seq_len][2][stat_dim]
                                              (if options['batch_norm'] and
                                               in inference mode)
                                              (absolute time index)
                            tt.alloc(0.)      (if not above or stat_dim == 0)
        Returns:
            tuple of 
                s_output_tbi, prev_state_update, s_stat_tsl
            where prev_state_update =
                (v_prev_state_bk, state[s_next_prev_idx]) (if state_dim > 0)
                None                                      (if state_dim == 0)
            and s_stat_tsl =
                bn_train stats (if options['batch_norm'] and in training mode)
                               (relative time index)
                None           (if not above or stat_dim == 0)
        
        - If state_dim > 0, v_prev_state_bk holds the state right before
          the 0-th time index of this time step
        - If state_dim > 0, step function should use tt.switch to pick
              states from (time index - 1)  (if time index > 0)
              s_init_state_k                (else)
          as the state values carried from (time index - 1)
        - If stat_dim > 0 and in inference mode, use v_stat_Tsl[time] as
          normalization statistics (handled automatically by bn function)
        - Receiving side should check if prev_state_update or s_stat_tsl
          is None or not
        """
        pass
    
    def pfx(self, s):
        return '%s_%s' % (self.name, s)

    def bn(self, x_bi, g_i, b_i, stat_Tsl, time, max_time, n, stride):
        """
        Inputs:
            x_bi        [batch_size][i]
            g_i, b_i    [i]                     (gamma & beta parameters)
            stat_Tsl    tt.alloc(0.)            (training mode)
                        [max_time + 1][2][l]    (inference mode)
            time        int32 symbolic scalar   (absolute time index)
            max_time    plain int
            n, stride   plain int
        Returns:
            y_bi        [batch_size][i] (normalized & affine transformed)
            stat_si     [2][i]          (save and use for inference)
        where
            stat_si[0] = mean_i 
            stat_si[1] = var_i
        """
        if b_i is None:
            b_i = tt.zeros_like(g_i)
        if stat_Tsl.ndim == 0: # training
            # out, mean, 1/sqrt(var + e); all in x_bi.ndim dimensions
            y_bi, m, s = cu.dnn_batch_normalization_train \
                             (x_bi, g_i[None, :], b_i[None, :],
                              epsilon = self.bn_eps)
            stat_si = tt.stack([m.flatten(),
                                tt.sqr(tt.inv(s.flatten())) - self.bn_eps])
        else: # inference
            stat_si = cut1(stat_Tsl[tt.minimum(time, max_time)], n, stride)
            y_bi = cu.dnn_batch_normalization_test \
                       (x_bi, g_i[None, :], b_i[None, :],
                        stat_si[0 : 1, :], stat_si[1 : 2, :],
                        epsilon = self.bn_eps)
        return y_bi, stat_si

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
    
    def setup_clock_graph(self, s_time_t, t_i, s_i):
        time_t = tt.cast(s_time_t, 'float32')
        phi_ti = tt.mod(time_t[:, :, None] - s_i, t_i) / t_i # broadcasts
        r_on  = self.clock_r_on
        alpha = self.clock_leak_rate
        return tt.switch(phi_ti < r_on / 2., 2. * phi_ti / r_on,
               tt.switch(phi_ti < r_on     , 2. - 2. * phi_ti / r_on,
                                             alpha * phi_ti))
    

class FCLayer(Layer):
    def add_param(self, params, n_in, n_out, options, **kwargs):
        # Pass, e.g., act = 'lambda x: tt.tanh(x)' as last argument
        assert callable(eval(kwargs['act']))
        self._act = eval(kwargs['act'])

        params[self.pfx('W')] = unif_weight(options, n_in, n_out)
        params[self.pfx('b')] = unif_weight(options, n_out)

        self.use_res_gate = options['residual_gate'] and n_in == n_out
        if self.use_res_gate:
            params[self.pfx('rg_k')] = -1. * np.ones(n_out).astype('float32')
        
        return 0, 0 # state [], stat []

    def setup_graph(self, s_below_tbj, s_time_t, s_next_prev_idx,
                    v_params, v_prev_state_bk, v_init_state_k, v_stat_Tsl):
        v_param = lambda name: v_params[self.pfx(name)]

        h_tbi = self._act(tt.dot(s_below_tbj, v_param('W')) + v_param('b'))
        
        if not self.use_res_gate:
            out_tbi = h_tbi
        else:
            g_i = tt.nnet.sigmoid(v_param('rg_k'))
            out_tbi = g_i * h_tbi + (1. - g_i) * s_below_tbj

        return out_tbi, None, None # no state update, no statistics


class LSTMLayer(Layer):
    def add_param(self, params, n_in, n_out, options, **kwargs):
        self.n_steps        = options['window_size']
        self.n_out          = n_out
        self.use_peephole   = options['lstm_peephole']
        self.use_batch_norm = options['batch_norm']
        self.use_input_bias = not self.use_batch_norm
        self.max_time_idx   = options['seq_len'] - 1
        self.use_clock      = options['learn_clock_params']
        self.use_res_gate   = options['residual_gate'] and n_in == n_out
        self.unroll_scan    = options['unroll_scan']

        # input to (i, f, c, o) [n_in][4 * n_out]
        params[self.pfx('W')] = np.concatenate \
                                    ([unif_weight(options, n_in, n_out) \
                                      for _ in range(4)], axis = 1)
        if self.use_input_bias:
            params[self.pfx('b')] = unif_weight(options, 4 * n_out)
        # hidden to (i, f, c, o) [n_out][4 * n_out]
        params[self.pfx('U')] = np.concatenate \
                                    ([unif_weight(options, n_out, n_out) \
                                      for _ in range(4)], axis = 1)
        
        if self.use_peephole:
            params[self.pfx('p')] = unif_weight(options, 3 * n_out)

        if self.use_batch_norm:        
            params[self.pfx('bn_g')] = (0.1 # as suggested in arXiv:1603.09025
                                        * np.ones(9 * n_out).astype('float32'))
            params[self.pfx('bn_b')] = unif_weight(options, 5 * n_out)

        if self.use_clock:
            self.add_clock_params(params, n_out, options)

        if self.use_res_gate:
            params[self.pfx('rg_k')] = -1. * np.ones(n_out).astype('float32')

        # state [h[1], c[1]], stat [x[4], h[4], c[1]] (if applicable)
        return 2 * n_out, (9 * n_out if self.use_batch_norm else 0)

    def setup_graph(self, s_below_tbj, s_time_t, s_next_prev_idx,
                    v_params, v_prev_state_bk, v_init_state_k, v_stat_Tsl):
        v_param  = lambda name: v_params[self.pfx(name)]
        n_out    = self.n_out
        max_time = self.max_time_idx

        W_j4i  = v_param('W')
        b_4i   = v_param('b') if self.use_input_bias else 0.
        x_tb4i = tt.dot(s_below_tbj, W_j4i) + b_4i

        init_hc_2i = v_init_state_k
        U_i4i      = v_param('U')
        p_3i       = v_param('p') if self.use_peephole else \
                     tt.zeros(3 * n_out).astype('float32')
        non_seqs = [init_hc_2i, U_i4i, p_3i]

        mask_ti = self.setup_clock_graph \
                      (s_time_t, v_param('clk_t'), v_param('clk_s')) \
                  if self.use_clock else \
                  tt.ones((self.n_steps, 1), dtype = 'float32')

        if self.use_batch_norm:
            gx_4i = cut0(v_param('bn_g'), 0, 4 * n_out) # g[0, 4)
            bx_4i = cut0(v_param('bn_b'), 0, 4 * n_out) # b[0, 4)

            gh_4i = cut0(v_param('bn_g'), 1, 4 * n_out) # g[4, 8)

            gc_i  = cut0(v_param('bn_g'), 8, 1 * n_out) # g[8, 9)
            bc_i  = cut0(v_param('bn_b'), 4, 1 * n_out) # b[4, 5)

            non_seqs += [gx_4i, bx_4i, gh_4i, gc_i, bc_i, v_stat_Tsl]
        bn_training = self.use_batch_norm and v_stat_Tsl.ndim == 0

        def step(x_b4i, time, mask_i, prev_hc_b2i, *args):
            prev_hc_b2i = tt.switch(time > 0, prev_hc_b2i, init_hc_2i)
            prev_h_bi = cut1(prev_hc_b2i, 0, n_out)
            prev_c_bi = cut1(prev_hc_b2i, 1, n_out)
            
            h_b4i = tt.dot(prev_h_bi, U_i4i)
            if self.use_batch_norm:
                x_b4i, stat_x_s4i = self.bn(x_b4i, gx_4i, bx_4i, v_stat_Tsl,
                                            time, max_time, 0, 4 * n_out)
                h_b4i, stat_h_s4i = self.bn(h_b4i, gh_4i, None , v_stat_Tsl,
                                            time, max_time, 1, 4 * n_out)
            preact_b4i = x_b4i + h_b4i

            i_bi = tt.nnet.sigmoid(cut1(preact_b4i, 0, n_out)
                                 + cut0(p_3i, 0, n_out) * prev_c_bi)
            f_bi = tt.nnet.sigmoid(cut1(preact_b4i, 1, n_out)
                                 + cut0(p_3i, 1, n_out) * prev_c_bi)

            c_bi = (i_bi * tt.tanh(cut1(preact_b4i, 2, n_out))
                    + f_bi * prev_c_bi)
            if self.use_clock:
                c_bi = mask_i * c_bi + (1. - mask_i) * prev_c_bi

            o_bi = tt.nnet.sigmoid(cut1(preact_b4i, 3, n_out)
                                 + cut0(p_3i, 2, n_out) * c_bi)
            
            ct_bi = c_bi
            if self.use_batch_norm:
                ct_bi, stat_c_si = self.bn(ct_bi, gc_i, bc_i, v_stat_Tsl,
                                           time, max_time, 8, 1 * n_out)
            h_bi = o_bi * tt.tanh(ct_bi)
            if self.use_clock:
                h_bi = mask_i * h_bi + (1. - mask_i) * prev_h_bi 

            return (tt.concatenate([h_bi, c_bi], axis = 1),
                    (tt.concatenate([stat_x_s4i, stat_h_s4i, stat_c_si],
                                    axis = 1) \
                     if bn_training else tt.alloc(0.)))

        # hc_stat = [hc_tb2i, stat_tsl]
        if not self.unroll_scan:
            hc_stat, _ = th.scan(step,
                                 sequences     = [x_tb4i, s_time_t, mask_ti],
                                 outputs_info  = [v_prev_state_bk, None],
                                 non_sequences = non_seqs,
                                 n_steps       = self.n_steps,
                                 name          = self.pfx('scan'),
                                 strict        = True)
        else:
            hc_list   = [v_prev_state_bk]
            stat_list = [None]
            for t in range(self.n_steps):
                hc_b2i, stat_sl = step(below_tb4i[t], s_time_t[t], mask_ti[t],
                                       hc_list[-1], *non_seqs)
                hc_list.append(hc_b2i)
                stat_list.append(stat_sl)
            hc_stat = [tt.stack(hc_list[1 :]), tt.stack(stat_list[1 :])]

        h_tbi = hc_stat[0][:, :, : n_out]
        if not self.use_res_gate:
            out_tbi = h_tbi
        else:
            g_i = tt.nnet.sigmoid(v_param('rg_k'))
            out_tbi = g_i * h_tbi + (1. - g_i) * s_below_tbj

        return (out_tbi,
                (v_prev_state_bk, hc_stat[0][s_next_prev_idx]),
                (hc_stat[1] if bn_training else None))


class GRULayer(Layer):
    def add_param(self, params, n_in, n_out, options, **kwargs):
        self.n_steps        = options['window_size']
        self.n_out          = n_out
        self.use_batch_norm = options['batch_norm']
        self.use_input_bias = not self.use_batch_norm
        self.max_time_idx   = options['seq_len'] - 1
        self.use_clock      = options['learn_clock_params']
        self.use_res_gate   = options['residual_gate'] and n_in == n_out
        self.unroll_scan    = options['unroll_scan']

        # input to (r, u, c) [n_in][3 * n_out]
        params[self.pfx('W')] = np.concatenate \
                                    ([unif_weight(options, n_in, n_out) \
                                      for _ in range(3)], axis = 1)
        if self.use_input_bias:
            params[self.pfx('b')] = unif_weight(options, 3 * n_out)
        # hidden to (r, u, c) [n_out][3 * n_out]
        params[self.pfx('U')] = np.concatenate \
                                    ([unif_weight(options, n_out, n_out) \
                                      for _ in range(3)], axis = 1)
                
        if self.use_batch_norm:
            params[self.pfx('bn_g')] = (0.1 # as suggested in arXiv:1603.09025
                                        * np.ones(6 * n_out).astype('float32'))
            params[self.pfx('bn_b')] = unif_weight(options, 4 * n_out)
        
        if self.use_clock:
            self.add_clock_params(params, n_out, options)

        if self.use_res_gate:
            params[self.pfx('rg_k')] = -1. * np.ones(n_out).astype('float32')
        
        # state [h[1]], stat [x[2], h[2], xc[1], hc[1]] (if applicable)
        return n_out, (6 * n_out if self.use_batch_norm else 0)

    def setup_graph(self, s_below_tbj, s_time_t, s_next_prev_idx,
                    v_params, v_prev_state_bk, v_init_state_k, v_stat_Tsl):
        v_param  = lambda name: v_params[self.pfx(name)]
        n_out    = self.n_out
        max_time = self.max_time_idx

        W_j3i  = v_param('W')
        b_3i   = v_param('b') if self.use_input_bias else 0.
        x_tb3i = tt.dot(s_below_tbj, W_j3i) + b_3i

        init_h_i = v_init_state_k
        U_i3i    = v_param('U')
        non_seqs = [init_h_i, U_i3i]

        mask_ti = self.setup_clock_graph \
                      (s_time_t, v_param('clk_t'), v_param('clk_s')) \
                  if self.use_clock else \
                  tt.ones((self.n_steps, 1), dtype = 'float32')
        
        if self.use_batch_norm:
            gx_2i = cut0(v_param('bn_g'), 0, 2 * n_out) # g[0, 2)
            bx_2i = cut0(v_param('bn_b'), 0, 2 * n_out) # b[0, 2)

            gh_2i = cut0(v_param('bn_g'), 1, 2 * n_out) # g[2, 4)

            gxc_i = cut0(v_param('bn_g'), 4, 1 * n_out) # g[4, 5)
            bxc_i = cut0(v_param('bn_b'), 2, 1 * n_out) # b[2, 3)

            ghc_i = cut0(v_param('bn_g'), 5, 1 * n_out) # g[5, 6)
            bhc_i = cut0(v_param('bn_b'), 3, 1 * n_out) # b[3, 4)

            non_seqs += [gx_2i, bx_2i, gh_2i, gxc_i, bxc_i, ghc_i, bhc_i,
                         v_stat_Tsl]
        bn_training = self.use_batch_norm and v_stat_Tsl.ndim == 0

        def step(x_b3i, time, mask_i, prev_h_bi, *args):
            prev_h_bi = tt.switch(time > 0, prev_h_bi, init_h_i)
            
            x_b2i = cut1(x_b3i, 0, 2 * n_out)
            h_b2i = tt.dot(prev_h_bi, cut1(U_i3i, 0, 2 * n_out))
            if self.use_batch_norm:
                x_b2i, stat_x_s2i = self.bn(x_b2i, gx_2i, bx_2i, v_stat_Tsl,
                                            time, max_time, 0, 2 * n_out)
                h_b2i, stat_h_s2i = self.bn(h_b2i, gh_2i, None , v_stat_Tsl,
                                            time, max_time, 1, 2 * n_out)
            preact_b2i = x_b2i + h_b2i

            r_bi = tt.nnet.sigmoid(cut1(preact_b2i, 0, n_out))
            u_bi = tt.nnet.sigmoid(cut1(preact_b2i, 1, n_out))

            xc_bi = cut1(x_b3i, 2, 1 * n_out)
            hc_bi = tt.dot(prev_h_bi, cut1(U_i3i, 2, 1 * n_out))
            if self.use_batch_norm:
                xc_bi, stat_xc_si = self.bn(xc_bi, gxc_i, bxc_i, v_stat_Tsl,
                                            time, max_time, 4, 1 * n_out)
                hc_bi, stat_hc_si = self.bn(hc_bi, ghc_i, bhc_i, v_stat_Tsl,
                                            time, max_time, 5, 1 * n_out)
            c_bi = tt.tanh(xc_bi + r_bi * hc_bi)

            h_bi = (1. - u_bi) * prev_h_bi + u_bi * c_bi
            if self.use_clock:
                h_bi = mask_i * h_bi + (1. - mask_i) * prev_h_bi

            return h_bi, (tt.concatenate([stat_x_s2i, stat_h_s2i,
                                          stat_xc_si, stat_hc_si], axis = 1) \
                          if bn_training else tt.alloc(0.))

        # h_stat = [h_tbi, stat_tsl]
        if not self.unroll_scan:
            h_stat, _ = th.scan(step,
                                sequences     = [x_tb3i, s_time_t, mask_ti],
                                outputs_info  = [v_prev_state_bk, None],
                                non_sequences = non_seqs,
                                n_steps       = n_steps,
                                name          = self.pfx('scan'),
                                strict        = True)
        else:
            h_list    = [v_prev_state_bk]
            stat_list = [None]
            for t in range(self.n_steps):
                h_bi, stat_sl = step(x_tb3i[t], s_time_t[t], mask_ti[t],
                                     h_list[-1], *non_seqs)
            h_stat = [tt.stack(h_list[1 :]), tt.stack(stat_list[1 :])]
        
        h_tbi = h_stat[0]
        if not self.use_res_gate:
            out_tbi = h_tbi
        else:
            g_i = tt.nnet.sigmoid(v_param('rg_k'))
            out_tbi = g_i * h_tbi + (1. - g_i) * s_below_tbj

        return (out_tbi,
                (v_prev_state_bk, h_tbi[s_next_prev_idx]),
                (hc_stat[1] if bn_training else None))

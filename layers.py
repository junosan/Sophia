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
from utils import unif_weight

class Layer():
    __metaclass__ = ABCMeta

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

    def slice1(self, x_i, n, stride):
        return x_i[n * stride : (n + 1) * stride]

    def slicei(self, x_bi, n, stride):
        return x_bi[:, n * stride : (n + 1) * stride]
    
    def sliceb(self, x_bi, n, stride):
        return x_bi[n * stride : (n + 1) * stride, :]

    def layer_norm(self, x_bi, s_i, b_i):
        y_bi = (x_bi - x_bi.mean(1)[:, None]) / tt.sqrt(x_bi.var(1)[:, None]
                                                        + 1e-5)
        return s_i[None, :] * y_bi + b_i[None, :]
    
    def layer_norm_lambdas(self, s_ci, b_ci, c, stride):
        """
            s_ci, b_ci  symbolic    [c * stride]
        """
        return [(lambda x_bi, i = i: \
                      self.layer_norm(x_bi, self.slice1(s_ci, i, stride),
                                            self.slice1(b_ci, i, stride))) \
                 for i in range(c)] # i = i part due to Python 2.7 limitations

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

        # In Fractal, W & b were both ~ Uniform [-0.02, 0.02)
        params[self.pfx('W')] = unif_weight(options, n_in, n_out)
        params[self.pfx('b')] = unif_weight(options, n_out)

        self.skip_connection = options['skip_connection'] and n_in == n_out
        if self.skip_connection:
            params[self.pfx('sc_k')] = -1. * np.ones(n_out).astype('float32')
        
        return 0 # no state variables

    def setup_graph(self, s_below_tbj, s_time_tb, s_next_prev_idx,
                    v_params, v_prev_state_bk, v_init_state_k):
        v_param = lambda name: v_params[self.pfx(name)]

        h_tbi = self._act(tt.dot(s_below_tbj, v_param('W')) + v_param('b'))
        
        if not self.skip_connection:
            out_tbi = h_tbi
        else:
            g_i = tt.nnet.sigmoid(v_param('sc_k'))
            out_tbi = g_i * h_tbi + (1. - g_i) * s_below_tbj

        return out_tbi, None # no prev_state_update


class LSTMLayer(Layer):
    def add_param(self, params, n_in, n_out, options, **kwargs):
        self.n_steps = options['window_size']
        self.unroll_scan = options['unroll_scan']
        self.n_out = n_out

        # In Fractal, W, b, U, ph were all ~ Uniform [-0.02, 0.02)
        # input to (i, f, o, c) [n_in][4 * n_out]
        params[self.pfx('W')] = np.concatenate \
                                    ([unif_weight(options, n_in, n_out) \
                                      for _ in range(4)], axis = 1)
        params[self.pfx('b')] = unif_weight(options, 4 * n_out)
        # hidden to (i, f, o, c) [n_out][4 * n_out]
        params[self.pfx('U')] = np.concatenate \
                                    ([unif_weight(options, n_out, n_out) \
                                      for _ in range(4)], axis = 1)
        
        self.use_peephole = options['lstm_peephole']
        if self.use_peephole:
            params[self.pfx('p')] = unif_weight(options, 3 * n_out)

        self.use_layer_norm = options['layer_norm']
        if self.use_layer_norm:
            params[self.pfx('ln_s')] = np.ones (4 * n_out).astype('float32')
            params[self.pfx('ln_b')] = np.zeros(4 * n_out).astype('float32')
        
        self.skip_connection = options['skip_connection'] and n_in == n_out
        if self.skip_connection:
            params[self.pfx('sc_k')] = -1. * np.ones(n_out).astype('float32')

        self.learn_clock_params = options['learn_clock_params']
        if self.learn_clock_params:
            self.add_clock_params(params, n_out, options)
        
        return 2 * n_out # h, c

    def setup_graph(self, s_below_tbj, s_time_tb, s_next_prev_idx,
                    v_params, v_prev_state_bk, v_init_state_k):
        v_param = lambda name: v_params[self.pfx(name)]
        n_out   = self.n_out

        if v_init_state_k is None:
            init_h_i = tt.zeros(n_out, dtype = 'float32')
            init_c_i = tt.zeros(n_out, dtype = 'float32')
        else:
            init_h_i = self.slice1(v_init_state_k, 0, n_out)
            init_c_i = self.slice1(v_init_state_k, 1, n_out)

        W_j4i = v_param('W')
        b_4i  = v_param('b')
        U_i4i = v_param('U')
        non_seqs = [init_h_i, init_c_i, U_i4i]

        if not self.use_peephole:
            p = [0.] * 3
        else:
            p = [self.slice1(v_param('p'), i, n_out) for i in range(3)]
            non_seqs.append(v_param('p'))

        if not self.use_layer_norm:
            l = [lambda x_bi: x_bi] * 4
        else:
            l = self.layer_norm_lambdas(v_param('ln_s'),
                                        v_param('ln_b'), 4, n_out)
            non_seqs.extend([v_param('ln_s'), v_param('ln_b')])

        if not self.learn_clock_params:
            mask_tbi = tt.ones((self.n_steps, 1, 1), dtype = 'float32')
        else:
            mask_tbi = self.setup_clock_graph(s_time_tb, v_param('clk_t'),
                                                         v_param('clk_s'))

        below_tb4i = tt.dot(s_below_tbj, W_j4i) + b_4i

        def step(below_b4i, time_b, mask_bi, prev_h_bi, prev_c_bi, *args):
            prev_h_bi = tt.switch(time_b[:, None] > 0., prev_h_bi, init_h_i)
            prev_c_bi = tt.switch(time_b[:, None] > 0., prev_c_bi, init_c_i)
            
            preact_b4i = below_b4i + tt.dot(prev_h_bi, U_i4i) 

            i_bi = tt.nnet.sigmoid(l[0](self.slicei(preact_b4i, 0, n_out)
                                        + p[0] * prev_c_bi))
            f_bi = tt.nnet.sigmoid(l[1](self.slicei(preact_b4i, 1, n_out)
                                        + p[1] * prev_c_bi))

            c_bi = (i_bi * tt.tanh(l[2](self.slicei(preact_b4i, 2, n_out)))
                    + f_bi * prev_c_bi)
            c_bi = mask_bi * c_bi + (1. - mask_bi) * prev_c_bi

            o_bi = tt.nnet.sigmoid(l[3](self.slicei(preact_b4i, 3, n_out)
                                        + p[2] * c_bi))
            h_bi = o_bi * tt.tanh(c_bi)
            h_bi = mask_bi * h_bi + (1. - mask_bi) * prev_h_bi 

            return h_bi, c_bi

        # hc = [h_tbi, c_tbi]
        if not self.unroll_scan:
            hc, _ = th.scan(step,
                      sequences     = [below_tb4i, s_time_tb, mask_tbi],
                      outputs_info  = [self.slicei(v_prev_state_bk, 0, n_out),
                                       self.slicei(v_prev_state_bk, 1, n_out)],
                      non_sequences = non_seqs,
                      n_steps       = self.n_steps,
                      name          = self.pfx('scan'),
                      strict        = True)
        else:
            h_list = [self.slicei(v_prev_state_bk, 0, n_out)]
            c_list = [self.slicei(v_prev_state_bk, 1, n_out)]
            for t in range(self.n_steps):
                h_bi, c_bi = step(below_tb4i[t], s_time_tb[t], mask_tbi[t],
                                  h_list[-1], c_list[-1])
                h_list.append(h_bi)
                c_list.append(c_bi)
            hc = [tt.stack(h_list[1 :]), tt.stack(c_list[1 :])]

        if not self.skip_connection:
            out_tbi = hc[0]
        else:
            g_i = tt.nnet.sigmoid(v_param('sc_k'))
            out_tbi = g_i * hc[0] + (1. - g_i) * s_below_tbj

        return out_tbi, (v_prev_state_bk,
                         tt.concatenate([hc[0][s_next_prev_idx],
                                         hc[1][s_next_prev_idx]], axis = 1))


class GRULayer(Layer):
    def add_param(self, params, n_in, n_out, options, **kwargs):
        self.n_steps = options['window_size']
        self.unroll_scan = options['unroll_scan']
        self.n_out = n_out

        # input to (r, u) [n_in][2 * n_out]
        params[self.pfx('W')] = np.concatenate \
                                    ([unif_weight(options, n_in, n_out) \
                                      for _ in range(2)], axis = 1)
        params[self.pfx('b')] = unif_weight(options, 2 * n_out)
        # hidden to (r, u) [n_out][2 * n_out]
        params[self.pfx('U')] = np.concatenate \
                                    ([unif_weight(options, n_out, n_out) \
                                      for _ in range(2)], axis = 1)
        # input to hidden [n_in][n_out]
        params[self.pfx('Wh')] = unif_weight(options, n_in, n_out)
        params[self.pfx('bh')] = unif_weight(options, n_out)
        # hidden to hidden [n_out][n_out]
        params[self.pfx('Uh')] = unif_weight(options, n_out, n_out)
        
        self.use_layer_norm = options['layer_norm']
        if self.use_layer_norm:
            params[self.pfx('ln_s')] = np.ones (3 * n_out).astype('float32')
            params[self.pfx('ln_b')] = np.zeros(3 * n_out).astype('float32')

        self.skip_connection = options['skip_connection'] and n_in == n_out
        if self.skip_connection:
            params[self.pfx('sc_k')] = -1. * np.ones(n_out).astype('float32')

        self.learn_clock_params = options['learn_clock_params']
        if self.learn_clock_params:
            self.add_clock_params(params, n_out, options)
        
        return n_out # h

    def setup_graph(self, s_below_tbj, s_time_tb, s_next_prev_idx,
                    v_params, v_prev_state_bk, v_init_state_k):
        v_param = lambda name: v_params[self.pfx(name)]
        n_out   = self.n_out

        if v_init_state_k is None:
            init_h_i = tt.zeros(n_out, dtype = 'float32')
        else:
            init_h_i = v_init_state_k
        
        W_j2i = v_param('W')
        b_2i  = v_param('b')
        U_i2i = v_param('U')
        Wh_ji = v_param('Wh')
        bh_i  = v_param('bh')
        Uh_ii = v_param('Uh')
        non_seqs = [init_h_i, U_i2i, Uh_ii]

        if not self.use_layer_norm:
            l = [lambda x_bi: x_bi] * 3
        else:
            l = self.layer_norm_lambdas(v_param('ln_s'),
                                        v_param('ln_b'), 3, n_out)
            non_seqs.extend([v_param('ln_s'), v_param('ln_b')])

        if not self.learn_clock_params:
            mask_tbi = tt.ones((self.n_steps, 1, 1), dtype = 'float32')
        else:
            mask_tbi = self.setup_clock_graph(s_time_tb, v_param('clk_t'),
                                                         v_param('clk_s'))

        below_tb2i = tt.dot(s_below_tbj, W_j2i) + b_2i
        belowh_tbi = tt.dot(s_below_tbj, Wh_ji) + bh_i

        def step(below_b2i, belowh_bi, time_b, mask_bi, prev_h_bi, *args):
            prev_h_bi = tt.switch(time_b[:, None] > 0., prev_h_bi, init_h_i)
            
            preact_b2i = below_b2i + tt.dot(prev_h_bi, U_i2i)

            r_bi = tt.nnet.sigmoid(l[0](self.slicei(preact_b2i, 0, n_out)))
            u_bi = tt.nnet.sigmoid(l[1](self.slicei(preact_b2i, 1, n_out)))

            c_bi = tt.tanh(l[2](belowh_bi + r_bi * tt.dot(prev_h_bi, Uh_ii)))

            h_bi = (1. - u_bi) * prev_h_bi + u_bi * c_bi
            h_bi = mask_bi * h_bi + (1. - mask_bi) * prev_h_bi

            return h_bi

        if not self.unroll_scan:
            h_tbi, _ = th.scan(step,
                               sequences     = [below_tb2i, belowh_tbi,
                                                s_time_tb, mask_tbi],
                               outputs_info  = v_prev_state_bk,
                               non_sequences = non_seqs,
                               n_steps       = n_steps,
                               name          = self.pfx('scan'),
                               strict        = True)
        else:
            h_list = [v_prev_state_bk]
            for t in range(self.n_steps):
                h_list.append(step(below_tb2i[t], belowh_tbi[t], s_time_tb[t],
                                   mask_tbi[t], h_list[-1]))
            h_tbi = tt.stack(h_list[1 :])

        if not self.skip_connection:
            out_tbi = h_tbi
        else:
            g_i = tt.nnet.sigmoid(v_param('sc_k'))
            out_tbi = g_i * h_tbi + (1. - g_i) * s_below_tbj

        return out_tbi, (v_prev_state_bk, h_tbi[s_next_prev_idx])

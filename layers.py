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
        Implements:
            Add this layer's parameters to params
            If learn_init_states, add init_i as well (recurrent layers only)
            It should have the same dimensions as the layer's internal state
            variables and its name must be $_init, matching $_prev below
        Inputs:
           &params  OrderedDict { str : np.ndarray }
            n_in    int
            n_out   int
            options OrderedDict { str : varies     }
            kwargs  dict        { str : varies     } (for any other settings)
        Returns:
            None
        """
        pass
    
    @abstractmethod
    def add_v_prev_state(self, v_prev_states):
        """
        Implements:
            Add this layer's v_prev_state_bi to v_prev_states (for recurrent)
            v_prev_state_bi should have the same dimensions as step function's
            output (i.e., $_init stacked batch_size times) and its name must
            be $_prev
        Inputs:
           &v_prev_states   OrderedDict { str : th.SharedVariable }
        Returns:
            None
        """
        pass
    
    @abstractmethod
    def setup_graph(self, s_below_tbj, s_time_t,
                    s_last_tap, v_params, v_prev_states):
        """
        Implements:
            Connect nodes below and global time node (if applicable) to outputs
            and prepare prev_state_update for the next time step
        Inputs:
            s_below_tbj     symbolic    [n_steps][batch_size][n_in]
            s_time_t        symbolic    [n_steps] (or None)
            s_last_tap      symbolic    int32 scalar (index)
            v_params        OrderedDict { str : th.SharedVariable }
            v_prev_states   OrderedDict { str : th.SharedVariable }
        Returns:
            tuple of 
                s_output_tbi, prev_state_update
            where prev_state_update = (v_prev_state, s_next_prev_state)
            with s_next_prev_state = v_prev_state       (if s_last_tap == -1)
                                     state[s_last_tap]  (otherwise)
            Receiving side should check if prev_state_update is None or not
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
    
    def setup_clock_graph(self, s_time_t, t_i, s_i):
        # mod & switch are elementwise
        # output dimensions: _ti
        phi_ti = tt.mod(s_time_t[:, None] - s_i, t_i) / t_i # note: broadcasts
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

        # In Fractal, W & b were both ~ Uniform [-0.02, 0.02)
        params[self.pfx('W')] = unif_weight(options, n_in, n_out)
        params[self.pfx('b')] = unif_weight(options, n_out)

        self.skip_connection = options['skip_connection'] and n_in == n_out
        if self.skip_connection:
            params[self.pfx('sc_k')] = -1. * np.ones(n_out).astype('float32')

    def add_v_prev_state(self, v_prev_states):
        pass
    
    def setup_graph(self, s_below_tbj, s_time_t,
                    s_last_tap, v_params, v_prev_states):
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
        self.n_out = n_out
        self.batch_size = options['batch_size']

        # In Fractal, W, b, U, ph were all ~ Uniform [-0.02, 0.02)
        # input to (i, f, o, c) [n_in][4 * n_out]
        params[self.pfx('W')] = np.concatenate \
                                    ([unif_weight(options, n_in, n_out),
                                      unif_weight(options, n_in, n_out),
                                      unif_weight(options, n_in, n_out),
                                      unif_weight(options, n_in, n_out)],
                                      axis = 1)
        params[self.pfx('b')] = unif_weight(options, 4 * n_out)
        # hidden to (i, f, o, c) [n_out][4 * n_out]
        params[self.pfx('U')] = np.concatenate \
                                    ([unif_weight(options, n_out, n_out),
                                      unif_weight(options, n_out, n_out),
                                      unif_weight(options, n_out, n_out),
                                      unif_weight(options, n_out, n_out)],
                                      axis = 1)
        
        self.use_peephole = options['lstm_peephole']
        if self.use_peephole:
            params[self.pfx('p')] = unif_weight(options, 3 * n_out)

        if options['learn_init_states']: # init_h, init_c
            params[self.pfx('init')] = np.zeros(2 * n_out).astype('float32')

        self.use_layer_norm = options['layer_norm']
        if self.use_layer_norm:
            params[self.pfx('ln_s')] = np.ones (4 * n_out).astype('float32')
            params[self.pfx('ln_b')] = np.zeros(4 * n_out).astype('float32')
        
        self.skip_connection = options['skip_connection'] and n_in == n_out
        if self.skip_connection:
            params[self.pfx('sc_k')] = -1. * np.ones(n_out).astype('float32')

        if options['learn_clock_params']:
            self.add_clock_params(params, n_out, options)
    
    def add_v_prev_state(self, v_prev_states):
        v_prev_states[self.pfx('prev')] \
            = th.shared(np.zeros((self.batch_size, 2 * self.n_out)) \
                        .astype('float32'), name = self.pfx('prev'))
    
    def setup_graph(self, s_below_tbj, s_time_t,
                    s_last_tap, v_params, v_prev_states):
        v_param = lambda name: v_params[self.pfx(name)]

        n_steps = s_below_tbj.shape[0] # symbolic   (can vary)
        n_out   = self.n_out           # int        (compile time constant)

        W_j4i = v_param('W')
        b_4i  = v_param('b')
        U_i4i = v_param('U')
        non_seqs = [U_i4i]

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

        if s_time_t is None: # not options['learn_clock_params']
            mask_ti = tt.alloc(1., n_steps, 1) # will broadcast
        else: 
            mask_ti = self.setup_clock_graph(s_time_t, v_param('clk_t'),
                                                       v_param('clk_s'))

        below_tb4i = tt.dot(s_below_tbj, W_j4i) + b_4i
        
        # (prev_h, prev_c)
        prev_state_b2i = v_prev_states[self.pfx('prev')]

        def step(mask_i, below_b4i, prev_h_bi, prev_c_bi, *args):
            preact_b4i = below_b4i + tt.dot(prev_h_bi, U_i4i) 

            i_bi = tt.nnet.sigmoid(l[0](self.slicei(preact_b4i, 0, n_out)
                                        + p[0] * prev_c_bi))
            f_bi = tt.nnet.sigmoid(l[1](self.slicei(preact_b4i, 1, n_out)
                                        + p[1] * prev_c_bi))

            c_bi = (i_bi * tt.tanh(l[2](self.slicei(preact_b4i, 2, n_out)))
                    + f_bi * prev_c_bi)
            c_bi = mask_i * c_bi + (1. - mask_i) * prev_c_bi

            o_bi = tt.nnet.sigmoid(l[3](self.slicei(preact_b4i, 3, n_out)
                                        + p[2] * c_bi))
            h_bi = o_bi * tt.tanh(c_bi)
            h_bi = mask_i * h_bi + (1. - mask_i) * prev_h_bi 

            return h_bi, c_bi

        # hc = [h_tbi, c_tbi]
        hc, _ = th.scan(step,
                     sequences     = [mask_ti, below_tb4i],
                     outputs_info  = [self.slicei(prev_state_b2i, 0, n_out),
                                      self.slicei(prev_state_b2i, 1, n_out)],
                     non_sequences = non_seqs,
                     n_steps       = n_steps,
                     name          = self.pfx('scan'),
                     strict        = True)
        
        next_prev_state = tt.switch(tt.eq(s_last_tap, -1), prev_state_b2i,
                              tt.concatenate([hc[0][s_last_tap],
                                              hc[1][s_last_tap]], axis = 1))

        if not self.skip_connection:
            out_tbi = hc[0]
        else:
            g_i = tt.nnet.sigmoid(v_param('sc_k'))
            out_tbi = g_i * hc[0] + (1. - g_i) * s_below_tbj

        return out_tbi, (v_prev_states[self.pfx('prev')], next_prev_state)


class GRULayer(Layer):
    def add_param(self, params, n_in, n_out, options, **kwargs):
        self.n_out = n_out
        self.batch_size = options['batch_size']

        # input to (r, u) [n_in][2 * n_out]
        params[self.pfx('W')] = np.concatenate \
                                    ([unif_weight(options, n_in, n_out),
                                      unif_weight(options, n_in, n_out)],
                                      axis = 1)
        params[self.pfx('b')] = unif_weight(options, 2 * n_out)
        # hidden to (r, u) [n_out][2 * n_out]
        params[self.pfx('U')] = np.concatenate \
                                    ([unif_weight(options, n_out, n_out),
                                      unif_weight(options, n_out, n_out)],
                                      axis = 1)
        # input to hidden [n_in][n_out]
        params[self.pfx('Wh')] = unif_weight(options, n_in, n_out)
        params[self.pfx('bh')] = unif_weight(options, n_out)
        # hidden to hidden [n_out][n_out]
        params[self.pfx('Uh')] = unif_weight(options, n_out, n_out)
        
        if options['learn_init_states']: # init_h
            params[self.pfx('init')] = np.zeros(n_out).astype('float32')
        
        self.use_layer_norm = options['layer_norm']
        if self.use_layer_norm:
            params[self.pfx('ln_s')] = np.ones (3 * n_out).astype('float32')
            params[self.pfx('ln_b')] = np.zeros(3 * n_out).astype('float32')

        self.skip_connection = options['skip_connection'] and n_in == n_out
        if self.skip_connection:
            params[self.pfx('sc_k')] = -1. * np.ones(n_out).astype('float32')

        if options['learn_clock_params']:
            self.add_clock_params(params, n_out, options)

    def add_v_prev_state(self, v_prev_states):
        v_prev_states[self.pfx('prev')] \
            = th.shared(np.zeros((self.batch_size, self.n_out)) \
                        .astype('float32'), name = self.pfx('prev'))

    def setup_graph(self, s_below_tbj, s_time_t,
                    s_last_tap, v_params, v_prev_states):
        v_param = lambda name: v_params[self.pfx(name)]

        n_steps = s_below_tbj.shape[0] # symbolic   (can vary)
        n_out   = self.n_out           # int        (compile time constant)

        W_j2i = v_param('W')
        b_2i  = v_param('b')
        U_i2i = v_param('U')
        Wh_ji = v_param('Wh')
        bh_i  = v_param('bh')
        Uh_ii = v_param('Uh')
        non_seqs = [U_i2i, Uh_ii]

        if not self.use_layer_norm:
            l = [lambda x_bi: x_bi] * 3
        else:
            l = self.layer_norm_lambdas(v_param('ln_s'),
                                        v_param('ln_b'), 3, n_out)
            non_seqs.extend([v_param('ln_s'), v_param('ln_b')])

        if s_time_t is None: # not options['learn_clock_params']
            mask_ti = tt.alloc(1., n_steps, 1) # will broadcast
        else: 
            mask_ti = self.setup_clock_graph(s_time_t, v_param('clk_t'),
                                                       v_param('clk_s'))

        below_tb2i = tt.dot(s_below_tbj, W_j2i) + b_2i
        belowh_tbi = tt.dot(s_below_tbj, Wh_ji) + bh_i

        def step(mask_i, below_b2i, belowh_bi, prev_h_bi, *args):
            preact_b2i = below_b2i + tt.dot(prev_h_bi, U_i2i)

            r_bi = tt.nnet.sigmoid(l[0](self.slicei(preact_b2i, 0, n_out)))
            u_bi = tt.nnet.sigmoid(l[1](self.slicei(preact_b2i, 1, n_out)))

            c_bi = tt.tanh(l[2](belowh_bi + r_bi * tt.dot(prev_h_bi, Uh_ii)))

            h_bi = (1. - u_bi) * prev_h_bi + u_bi * c_bi
            h_bi = mask_i * h_bi + (1. - mask_i) * prev_h_bi

            return h_bi

        h_tbi, _ = th.scan(step,
                       sequences     = [mask_ti, below_tb2i, belowh_tbi],
                       outputs_info  = v_prev_states[self.pfx('prev')],
                       non_sequences = non_seqs,
                       n_steps       = n_steps,
                       name          = self.pfx('scan'),
                       strict        = True)

        next_prev_state = tt.switch(tt.eq(s_last_tap, -1), 
                                    v_prev_states[self.pfx('prev')],
                                    h_tbi[s_last_tap])

        if not self.skip_connection:
            out_tbi = h_tbi
        else:
            g_i = tt.nnet.sigmoid(v_param('sc_k'))
            out_tbi = g_i * h_tbi + (1. - g_i) * s_below_tbj

        return out_tbi, (v_prev_states[self.pfx('prev')], next_prev_state)


class PILSTMLayer(Layer):
    def add_param(self, params, n_in, n_out, options, **kwargs):
        self.n_in  = n_in
        self.n_out = n_out
        self.batch_size = options['batch_size']

        # (p, i, h) to (u, c, o)
        params[self.pfx('Wp')] = np.concatenate \
                                    ([unif_weight(options, n_in, n_out) \
                                      for _ in range(3)], axis = 1)
        params[self.pfx('Wi')] = np.concatenate \
                                    ([unif_weight(options, n_in, n_out) \
                                      for _ in range(3)], axis = 1)
        params[self.pfx('U')]  = np.concatenate \
                                    ([unif_weight(options, n_out, n_out) \
                                      for _ in range(3)], axis = 1)
        # rho, (b_u, b_c, b_o); sig(rho) = decay rate of i
        params[self.pfx('rho')] = np.random.uniform(low = 0., high = 4.,
                                           size = (n_in)).astype('float32')
        params[self.pfx('b')]   = unif_weight(options, 3 * n_out)

        self.use_peephole = options['lstm_peephole']
        if self.use_peephole:
            params[self.pfx('p')] = unif_weight(options, 2 * n_out)

        if options['learn_init_states']: # init_h, init_c, init_i
            params[self.pfx('init')] = np.zeros \
                                           (2 * n_out + n_in).astype('float32')

        self.use_layer_norm = options['layer_norm']
        if self.use_layer_norm:
            params[self.pfx('ln_s')] = np.ones (3 * n_out).astype('float32')
            params[self.pfx('ln_b')] = np.zeros(3 * n_out).astype('float32')
        
        self.skip_connection = options['skip_connection'] and n_in == n_out
        if self.skip_connection:
            params[self.pfx('sc_k')] = -1. * np.ones(n_out).astype('float32')

        # cannot be clocked
    
    def add_v_prev_state(self, v_prev_states):
        v_prev_states[self.pfx('prev')] \
            = th.shared(np.zeros((self.batch_size,
                                  2 * self.n_out + self.n_in)) \
                        .astype('float32'), name = self.pfx('prev'))
    
    def setup_graph(self, s_below_tbj, s_time_t,
                    s_last_tap, v_params, v_prev_states):
        v_param = lambda name: v_params[self.pfx(name)]

        n_steps = s_below_tbj.shape[0] # symbolic   (can vary)
        n_in    = self.n_in            # int        (compile time constant)
        n_out   = self.n_out           # int        (compile time constant)
        
        p_tbj = s_below_tbj

        Wp_j3i = v_param('Wp')
        Wi_j3i = v_param('Wi')
        U_i3i  = v_param('U')
        rho_j  = tt.nnet.sigmoid(v_param('rho'))
        b_3i   = v_param('b')
        non_seqs = [Wi_j3i, U_i3i, v_param('rho')]

        if not self.use_peephole:
            p = [0.] * 2
        else:
            p = [self.slice1(v_param('p'), i, n_out) for i in range(2)]
            non_seqs.append(v_param('p'))

        if not self.use_layer_norm:
            l = [lambda x_bi: x_bi] * 3
        else:
            l = self.layer_norm_lambdas(v_param('ln_s'),
                                        v_param('ln_b'), 3, n_out)
            non_seqs.extend([v_param('ln_s'), v_param('ln_b')])

        preact_pb_tb3i = tt.dot(p_tbj, Wp_j3i) + b_3i
        
        # (prev_h, prev_c, prev_i); k = 2 * i + j
        prev_state_bk = v_prev_states[self.pfx('prev')]        

        def step(p_bj, preact_pb_b3i, prev_h_bi, prev_c_bi, prev_i_bj, *args):
            i_bj = rho_j * prev_i_bj + (1. - rho_j) * p_bj
            preact_tb3i = (preact_pb_b3i + tt.dot(i_bj, Wi_j3i)
                           + tt.dot(prev_h_bi, U_i3i))

            u_bi = tt.nnet.sigmoid(l[0](self.slicei(preact_tb3i, 0, n_out) 
                                        + p[0] * prev_c_bi))
            c_bi = (u_bi * tt.tanh(l[1](self.slicei(preact_tb3i, 1, n_out)))
                    + (1. - u_bi) * prev_c_bi)
            o_bi = tt.nnet.sigmoid(l[2](self.slicei(preact_tb3i, 2, n_out)
                                        + p[1] * c_bi))
            h_bi = o_bi * c_bi

            return h_bi, c_bi, i_bj
        
        # hci = [h_tbi, c_tbi, i_tbj]
        hci, _ = th.scan(step,
                     sequences     = [p_tbj, preact_pb_tb3i],
                     outputs_info  = [self.slicei(prev_state_bk, 0, n_out),
                                      self.slicei(prev_state_bk, 1, n_out),
                                      prev_state_bk[:, 2 * n_out :]],
                     non_sequences = non_seqs,
                     n_steps       = n_steps,
                     name          = self.pfx('scan'),
                     strict        = True)
        
        next_prev_state = tt.switch(tt.eq(s_last_tap, -1), prev_state_bk,
                              tt.concatenate([hci[0][s_last_tap],
                                              hci[1][s_last_tap],
                                              hci[2][s_last_tap]], axis = 1))
        
        if not self.skip_connection:
            out_tbi = hci[0]
        else:
            g_i = tt.nnet.sigmoid(v_param('sc_k'))
            out_tbi = g_i * hci[0] + (1. - g_i) * s_below_tbj
        
        return out_tbi, (v_prev_states[self.pfx('prev')], next_prev_state)


class DILSTMLayer(Layer):
    def add_param(self, params, n_in, n_out, options, **kwargs):
        self.n_out = n_out
        self.batch_size = options['batch_size']

        # input to (i, f, c, o, g) [n_in][5 * n_out]
        params[self.pfx('W')] = np.concatenate \
                                    ([unif_weight(options, n_in, n_out) \
                                      for _ in range(5)], axis = 1)
        params[self.pfx('b')] = unif_weight(options, 5 * n_out)
        # (c, d) to (i, f, c, o, g) [n_out][5 * n_out]
        params[self.pfx('U')] = np.concatenate \
                                    ([unif_weight(options, n_out, n_out) \
                                      for _ in range(5)], axis = 1)
        params[self.pfx('V')] = np.concatenate \
                                    ([unif_weight(options, n_out, n_out) \
                                      for _ in range(5)], axis = 1)

        assert 'lstm_peephole' not in options

        if options['learn_init_states']: # init_c, init_d
            params[self.pfx('init')] = np.zeros(2 * n_out).astype('float32')

        self.use_layer_norm = options['layer_norm']
        if self.use_layer_norm:
            params[self.pfx('ln_s')] = np.ones (5 * n_out).astype('float32')
            params[self.pfx('ln_b')] = np.zeros(5 * n_out).astype('float32')
        
        self.skip_connection = options['skip_connection'] and n_in == n_out
        if self.skip_connection:
            params[self.pfx('sc_k')] = -1. * np.ones(n_out).astype('float32')

        # cannot be clocked
    
    def add_v_prev_state(self, v_prev_states):
        v_prev_states[self.pfx('prev')] \
            = th.shared(np.zeros((self.batch_size, 2 * self.n_out)) \
                        .astype('float32'), name = self.pfx('prev'))
    
    def setup_graph(self, s_below_tbj, s_time_t,
                    s_last_tap, v_params, v_prev_states):
        v_param = lambda name: v_params[self.pfx(name)]

        n_steps = s_below_tbj.shape[0] # symbolic   (can vary)
        n_out   = self.n_out           # int        (compile time constant)

        W_j5i = v_param('W')
        b_5i  = v_param('b')
        U_i5i = v_param('U')
        V_i5i = v_param('V')
        non_seqs = [U_i5i, V_i5i]

        if not self.use_layer_norm:
            l = [lambda x_bi: x_bi] * 5
        else:
            l = self.layer_norm_lambdas(v_param('ln_s'),
                                        v_param('ln_b'), 5, n_out)
            non_seqs.extend([v_param('ln_s'), v_param('ln_b')])

        below_tb5i = tt.dot(s_below_tbj, W_j5i) + b_5i
        
        # (prev_c, prev_d)
        prev_state_b2i = v_prev_states[self.pfx('prev')]

        def step(below_b5i, prev_c_bi, prev_d_bi, *args):
            preact_b5i = (below_b5i + tt.dot(prev_c_bi, U_i5i)
                          + tt.dot(prev_d_bi, V_i5i))

            i_bi = tt.nnet.sigmoid(l[0](self.slicei(preact_b5i, 0, n_out)))
            f_bi = tt.nnet.sigmoid(l[1](self.slicei(preact_b5i, 1, n_out)))
            c_bi = (i_bi * tt.tanh(l[2](self.slicei(preact_b5i, 2, n_out)))
                    + f_bi * prev_c_bi)

            o_bi = tt.nnet.sigmoid(l[3](self.slicei(preact_b5i, 3, n_out)))
            h_bi = o_bi * tt.tanh(c_bi)
            
            g_bi = tt.nnet.sigmoid(l[4](self.slicei(preact_b5i, 4, n_out)))
            d_bi = g_bi * prev_d_bi + c_bi

            return h_bi, c_bi, d_bi

        # hcd = [h_tbi, c_tbi, d_tbi]
        hcd, _ = th.scan(step,
                     sequences     = [below_tb5i],
                     outputs_info  = [None,
                                      self.slicei(prev_state_b2i, 0, n_out),
                                      self.slicei(prev_state_b2i, 1, n_out)],
                     non_sequences = non_seqs,
                     n_steps       = n_steps,
                     name          = self.pfx('scan'),
                     strict        = True)
        
        next_prev_state = tt.switch(tt.eq(s_last_tap, -1), prev_state_b2i,
                              tt.concatenate([hcd[1][s_last_tap],
                                              hcd[2][s_last_tap]], axis = 1))

        if not self.skip_connection:
            out_tbi = hcd[0]
        else:
            g_i = tt.nnet.sigmoid(v_param('sc_k'))
            out_tbi = g_i * hcd[0] + (1. - g_i) * s_below_tbj

        return out_tbi, (v_prev_states[self.pfx('prev')], next_prev_state)

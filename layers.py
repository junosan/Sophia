#==========================================================================#
# Copyright (C) 2016 Hosang Yoon (hosangy@gmail.com) - All Rights Reserved #
# Unauthorized copying of this file, via any medium is strictly prohibited #
#                       Proprietary and confidential                       #
#==========================================================================#

"""
Layer classes; an abstract base class and specific layer types as derived classes
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
            Add this layer's v_prev_state_bi to v_prev_states (recurrent layers only)
            v_prev_state_bi should have the same dimensions as step function's output
            (i.e., $_init stacked batch_size times) and its name must be $_prev
        Inputs:
           &v_prev_states   OrderedDict { str : th.SharedVariable }
        Returns:
            None
        """
        pass
    
    @abstractmethod
    def setup_graph(self, s_below_tbj, s_time_t, v_params, v_prev_states):
        """
        Implements:
            Connect nodes below and global time node (if applicable) to outputs
            and prepare prev_state_update for the next time step
        Inputs:
            s_below_tbj     symbolic    [n_steps][batch_size][n_in] (even if n_steps == 1)
            s_time_t        symbolic    [n_steps]                   (even if n_steps == 1)
            v_params        OrderedDict { str : th.SharedVariable }
            v_prev_states   OrderedDict { str : th.SharedVariable } (recurrent layers only)
        Returns:
            tuple of 
                s_output_tbi, prev_state_update
            where prev_state_update = (v_prev_state, s_new_prev_state)
            Receiving side should check if prev_state_update is None before storing
        """
        pass
    
    def pfx(self, s):
        return '%s_%s' % (self.name, s)

    def slice1(self, x_i, n, stride):
        return x_i[n * stride : (n + 1) * stride]

    def slice2(self, x_bi, n, stride):
        return x_bi[:, n * stride : (n + 1) * stride]

    def layer_norm(self, x_bi, s_i, b_i):
        y_bi = (x_bi - x_bi.mean(1)[:, None]) / tt.sqrt(x_bi.var(1)[:, None] + 1e-5)
        return s_i[None, :] * y_bi + b_i[None, :]
    
    def add_clock_params(self, params, n_out, options):
        # t : period ~ exp(Uniform(lo, hi))
        # s : shift  ~ Uniform(0, t)
        params[self.pfx('t')] = np.exp(np.random.uniform(low  = options['clock_t_exp_lo'],
                                                         high = options['clock_t_exp_hi'],
                                                         size = (n_out))).astype('float32')
        params[self.pfx('s')] = params[self.pfx('t')] * \
                                np.random.uniform(size = (n_out)).astype('float32')
        self.clock_r_on      = options['clock_r_on']
        self.clock_leak_rate = options['clock_leak_rate']
    
    def setup_clock_graph(self, s_time_t, t_i, s_i):
        r_on  = self.clock_r_on
        alpha = self.clock_leak_rate
        # mod & switch are elementwise
        phi_ti = tt.mod(s_time_t[:, None] - s_i, t_i) / t_i # note: broadcasts
        return tt.switch(phi_ti < r_on / 2., 2. * phi_ti / r_on, # output dimensions: _ti
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

    def add_v_prev_state(self, v_prev_states):
        pass
    
    def setup_graph(self, s_below_tbj, s_time_t, v_params, v_prev_states):
        return self._act(tt.dot(s_below_tbj, v_params[self.pfx('W')]) + \
                         v_params[self.pfx('b')]), None # no prev_state_update


class LSTMLayer(Layer):
    def add_param(self, params, n_in, n_out, options, **kwargs):
        self.n_out = n_out
        self.batch_size = options['batch_size']
        self.use_peephole = options['unit_peephole']
        self.use_layer_norm = options['layer_norm']

        # In Fractal, W, b, U, ph were all ~ Uniform [-0.02, 0.02)
        # input to (i, f, o, c) [n_in][4 * n_out]
        params[self.pfx('W')] = np.concatenate([unif_weight(options, n_in, n_out),
                                                unif_weight(options, n_in, n_out),
                                                unif_weight(options, n_in, n_out),
                                                unif_weight(options, n_in, n_out)], axis = 1)
        params[self.pfx('b')] = unif_weight(options, 4 * n_out)
        # hidden to (i, f, o, c) [n_out][4 * n_out]
        params[self.pfx('U')] = np.concatenate([unif_weight(options, n_out, n_out),
                                                unif_weight(options, n_out, n_out),
                                                unif_weight(options, n_out, n_out),
                                                unif_weight(options, n_out, n_out)], axis = 1)
        
        if options['unit_peephole']:
            params[self.pfx('ph')] = unif_weight(options, 3 * n_out)

        if options['learn_init_states']:
            params[self.pfx('init')] = np.zeros(2 * n_out).astype('float32') # init_h, init_c

        if options['layer_norm']:
            params[self.pfx('ln_s')] = np.ones (4 * n_out).astype('float32')
            params[self.pfx('ln_b')] = np.zeros(4 * n_out).astype('float32')

        if options['learn_clock_params']:
            self.add_clock_params(params, n_out, options)
    
    def add_v_prev_state(self, v_prev_states):
        v_prev_states[self.pfx('prev')] = th.shared(np.zeros((self.batch_size, 2 * self.n_out)) \
                                                    .astype('float32'), name = self.pfx('prev'))
    
    def setup_graph(self, s_below_tbj, s_time_t, v_params, v_prev_states):
        v_param = lambda name: v_params[self.pfx(name)]

        W_j4i = v_param('W')
        b_4i  = v_param('b')
        U_i4i = v_param('U')
        non_seqs = [U_i4i]

        n_steps = s_below_tbj.shape[0] # can vary across th.function calls
        n_batch = s_below_tbj.shape[1]
        n_out   = U_i4i.shape[0]

        # options['unit_peephole']
        if not self.use_peephole:
            p = [0.] * 3
        else:
            p = [self.slice1(v_param('ph'), i, n_out) for i in range(3)]
            non_seqs.append(v_param('ph'))

        # options['layer_norm']
        if not self.use_layer_norm:
            thru = lambda x_bi: x_bi
            l = [thru] * 4
        else:
            l = [(lambda x_bi, i = i: \
                      self.layer_norm(x_bi, self.slice1(v_param('ln_s'), i, n_out),  \
                                            self.slice1(v_param('ln_b'), i, n_out))) \
                 for i in range(4)] # i = i part due to Python 2.7 limitations
            non_seqs.extend([v_param('ln_s'), v_param('ln_b')])

        # options['learn_clock_params']
        if s_time_t is None:
            mask_ti = tt.alloc(1., n_steps, 1) # will broadcast
        else: 
            mask_ti = self.setup_clock_graph(s_time_t, v_param('t'), v_param('s'))
        
        below_tb4i = tt.dot(s_below_tbj, W_j4i) + b_4i

        prev_state_b2i = v_prev_states[self.pfx('prev')]
        prev_h_bi = prev_state_b2i[:, 0     : n_out    ]
        prev_c_bi = prev_state_b2i[:, n_out : 2 * n_out]

        def step(mask_i, below_b4i, prev_h_bi, prev_c_bi, *args):
            preact_b4i = below_b4i + tt.dot(prev_h_bi, U_i4i) 

            i_bi = tt.nnet.sigmoid(l[0](self.slice2(preact_b4i, 0, n_out) + p[0] * prev_c_bi))
            f_bi = tt.nnet.sigmoid(l[1](self.slice2(preact_b4i, 1, n_out) + p[1] * prev_c_bi))

            c_bi = i_bi * tt.tanh (l[2](self.slice2(preact_b4i, 2, n_out))) + f_bi * prev_c_bi
            c_bi = mask_i * c_bi + (1. - mask_i) * prev_c_bi

            o_bi = tt.nnet.sigmoid(l[3](self.slice2(preact_b4i, 3, n_out) + p[2] * c_bi))
            h_bi = o_bi * tt.tanh(c_bi)
            h_bi = mask_i * h_bi + (1. - mask_i) * prev_h_bi 

            return h_bi, c_bi
        
        if n_steps == 1:
            ret = step(mask_ti[0], below_tb4i[0], prev_h_bi, prev_c_bi)
        else:
            ret, _ = th.scan(step, sequences     = [mask_ti, below_tb4i],
                                   outputs_info  = [prev_h_bi, prev_c_bi],
                                   non_sequences = non_seqs,
                                   n_steps       = n_steps,
                                   name          = self.pfx('scan'),
                                   strict        = True)
        
        h_tbi = ret[0].reshape((-1, n_batch, n_out))
        c_tbi = ret[1].reshape((-1, n_batch, n_out))

        return h_tbi, (v_prev_states[self.pfx('prev')], \
                       tt.concatenate([h_tbi[-1], c_tbi[-1]], axis = 1))


class GRULayer(Layer):
    def add_param(self, params, n_in, n_out, options, **kwargs):
        self.n_out = n_out
        self.batch_size = options['batch_size']
        self.use_layer_norm = options['layer_norm']

        # input to (r, u) [n_in][2 * n_out]
        params[self.pfx('W')] = np.concatenate([unif_weight(options, n_in, n_out),
                                                unif_weight(options, n_in, n_out)], axis = 1)
        params[self.pfx('b')] = unif_weight(options, 2 * n_out)
        # hidden to (r, u) [n_out][2 * n_out]
        params[self.pfx('U')] = np.concatenate([unif_weight(options, n_out, n_out),
                                                unif_weight(options, n_out, n_out)], axis = 1)
        # input to hidden [n_in][n_out]
        params[self.pfx('Wh')] = unif_weight(options, n_in, n_out)
        params[self.pfx('bh')] = unif_weight(options, n_out)
        # hidden to hidden [n_out][n_out]
        params[self.pfx('Uh')] = unif_weight(options, n_out, n_out)
        
        if options['learn_init_states']:
            params[self.pfx('init')] = np.zeros(n_out).astype('float32') # init_h
        
        if options['layer_norm']:
            params[self.pfx('ln_s')] = np.ones (3 * n_out).astype('float32')
            params[self.pfx('ln_b')] = np.zeros(3 * n_out).astype('float32')

        if options['learn_clock_params']:
            self.add_clock_params(params, n_out, options)

    def add_v_prev_state(self, v_prev_states):
        v_prev_states[self.pfx('prev')] = th.shared(np.zeros((self.batch_size, self.n_out)) \
                                                    .astype('float32'), name = self.pfx('prev'))

    def setup_graph(self, s_below_tbj, s_time_t, v_params, v_prev_states):
        v_param = lambda name: v_params[self.pfx(name)]

        W_j2i = v_param('W')
        b_2i  = v_param('b')
        U_i2i = v_param('U')
        Wh_ji = v_param('Wh')
        bh_i  = v_param('bh')
        Uh_ii = v_param('Uh')
        non_seqs = [U_i2i, Uh_ii]

        n_steps = s_below_tbj.shape[0] # can vary across th.function calls
        n_batch = s_below_tbj.shape[1]
        n_out   = U_i2i.shape[0]

        # options['layer_norm']
        if not self.use_layer_norm:
            thru = lambda x_bi: x_bi
            l = [thru] * 3
        else:
            l = [(lambda x_bi, i = i: \
                      self.layer_norm(x_bi, self.slice1(v_param('ln_s'), i, n_out),  \
                                            self.slice1(v_param('ln_b'), i, n_out))) \
                 for i in range(3)] # i = i part due to Python 2.7 limitations
            non_seqs.extend([v_param('ln_s'), v_param('ln_b')])

        # options['learn_clock_params']
        if s_time_t is None:
            mask_ti = tt.alloc(1., n_steps, 1) # will broadcast
        else: 
            mask_ti = self.setup_clock_graph(s_time_t, v_param('t'), v_param('s'))
        
        below_tb2i = tt.dot(s_below_tbj, W_j2i) + b_2i
        belowh_tbi = tt.dot(s_below_tbj, Wh_ji) + bh_i
        prev_h_bi  = v_prev_states[self.pfx('prev')]

        def step(mask_i, below_b2i, belowh_bi, prev_h_bi, *args):
            preact_b2i = below_b2i + tt.dot(prev_h_bi, U_i2i)

            r_bi = tt.nnet.sigmoid(l[0](self.slice2(preact_b2i, 0, n_out)))
            u_bi = tt.nnet.sigmoid(l[1](self.slice2(preact_b2i, 1, n_out)))

            c_bi = tt.tanh(l[2](belowh_bi + r_bi * tt.dot(prev_h_bi, Uh_ii)))

            h_bi = (1. - u_bi) * prev_h_bi + u_bi * c_bi
            h_bi = mask_i * h_bi + (1. - mask_i) * prev_h_bi

            return h_bi
        
        if n_steps == 1:
            ret = step(mask_ti[0], below_tb2i[0], belowh_tbi[0], prev_h_bi)
        else:
            ret, _ = th.scan(step, sequences     = [mask_ti, below_tb2i, belowh_tbi],
                                   outputs_info  = prev_h_bi,
                                   non_sequences = non_seqs,
                                   n_steps       = n_steps,
                                   name          = self.pfx('scan'),
                                   strict        = True)
        
        h_tbi = ret.reshape((-1, n_batch, n_out))

        return h_tbi, (v_prev_states[self.pfx('prev')], h_tbi[-1])

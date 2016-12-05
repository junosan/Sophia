#==========================================================================#
# Copyright (C) 2016 Hosang Yoon (hosangy@gmail.com) - All Rights Reserved #
# Unauthorized copying of this file, via any medium is strictly prohibited #
#                       Proprietary and confidential                       #
#==========================================================================#

"""
Optimizers
Implement the following signature:
    optimizer(s_lr, [v_param, ...], [v_grad, ...])
        -> [(v_optim_state, s_init_optim_state), ...],
           [(v_param, s_new_param), (v_optim_state, s_new_optim_state), ...]
Note:
    v_grads must be updated first before applying param_updates returned here
    Order between tuples in the updates list doesn't matter
"""

import numpy as np
import theano as th
import theano.tensor as tt

# Adam algorithm (arxiv:1412.6980)
# Other implementations:
#     https://github.com/Lasagne/Lasagne/blob/master/lasagne/updates.py
#     https://gist.github.com/Newmu/acb738767acb4788bac3
def adam(s_lr, v_params, v_grads):
    # Hyperparameters recommended in paper
    b1 = 0.9  
    b2 = 0.999
    e = 1e-8

    inits = []
    updates = []

    v_t = th.shared(np.float32(0.)) # scalar
    inits.append((v_t, tt.zeros_like(v_t)))

    s_new_t = v_t + 1.
    s_calibrated_lr = s_lr * tt.sqrt(1. - b2**(s_new_t)) / (1. - b1**(s_new_t))

    for v_param, v_grad in zip(v_params, v_grads):
        v_m = th.shared(v_param.get_value() * 0.) # th.shared(tt -> np 0-tensor)
        inits.append((v_m, tt.zeros_like(v_m)))
        
        v_v = th.shared(v_param.get_value() * 0.)
        inits.append((v_v, tt.zeros_like(v_v)))

        s_new_m = b1 * v_m + (1. - b1) * v_grad
        s_new_v = b2 * v_v + (1. - b2) * tt.sqr(v_grad)

        s_new_param = v_param - s_calibrated_lr * s_new_m / (tt.sqrt(s_new_v) + e)

        updates.append((v_m, s_new_m))
        updates.append((v_v, s_new_v))
        updates.append((v_param, s_new_param))

    updates.append((v_t, s_new_t))

    return inits, updates

# In the future, maybe add more optimizers
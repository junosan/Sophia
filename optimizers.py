#==========================================================================#
# Copyright (C) 2016 Hosang Yoon (hosangy@gmail.com) - All Rights Reserved #
# Unauthorized copying of this file, via any medium is strictly prohibited #
#                       Proprietary and confidential                       #
#==========================================================================#

"""
Optimizers
    optim_f_inits, optim_f_updates, s_forces \
        = force_func (options, s_lr    , v_grads )
    optim_u_inits, optim_u_updates + param_updates \
        = update_func(options, v_params, s_forces)

    to f_initialize_optimizer : optim_f_inits + optim_u_inits
    to f_update_v_params      : optim_f_updates + optim_u_updates
                                + param_updates
Note:
    v_params & v_grads must be lists of same shape & order
    v_grads must be updated before applying updates returned here
    Order between tuples in the updates list doesn't matter
"""

import numpy as np
import theano as th
import theano.tensor as tt
from utils import clip_elem

"""
Update functions to define increment for parameters, treating gradient as a
force along a potential hill with friction (unit mass & unit time step implied)
    v += - (1 - mu) v + external_force
where
    (1 - mu)      : dimensionless Stokes friction
    external_force: e.g., (-lr * grad) for vanilla_force

Implement the following signature:
    f(options, v_params, s_forces) -> optim_u_inits, updates
where updates includes both optim_u_updates + param_updates
"""

def sgd_update(options, v_params, s_forces):
    """
    Special case of momentum/nesterov with mu = 0
    (i.e., discrete version of critical damping)
        x += external_force
    """
    return [], [(v_param, v_param + s_force) \
                for v_param, s_force in zip(v_params, s_forces)]

def momentum_update(options, v_params, s_forces):
    """
    Naive update of momentum (velocity) with friction & external_force
        v = mu v + external_force
        x += v
    """
    mu = options['update_mu']

    inits   = []
    updates = []

    for v_param, s_force in zip(v_params, s_forces):
        v_v = th.shared(np.zeros_like(v_param.get_value()))
        inits.append((v_v, tt.zeros_like(v_v)))

        s_new_v = mu * v_v + s_force

        updates.append((v_param, v_param + s_new_v))
        updates.append((v_v, s_new_v))
    
    return inits, updates

def nesterov_update(options, v_params, s_forces):
    """
    Same as above, but with external_force at lookahead position
    (in terms of peeked-ahead params; arXiv:1212.0901 Eqs 6,7)
        x -= mu v
        v = mu v + external_force
        x += (1. + mu) v
    """
    mu = options['update_mu']

    inits   = []
    updates = []

    for v_param, s_force in zip(v_params, s_forces):
        v_v = th.shared(np.zeros_like(v_param.get_value()))
        inits.append((v_v, tt.zeros_like(v_v)))

        s_new_v = mu * v_v + s_force

        updates.append((v_param, v_param + (1. + mu) * s_new_v - mu * v_v))
        updates.append((v_v, s_new_v))

    return inits, updates


"""
Force functions to define increment for velocity
(i.e., external_force excluding friction)

Implement the following signature:
    f(options, s_lr, v_grads) -> optim_f_inits, optim_f_updates, s_forces
"""

def vanilla_force(options, s_lr, v_grads):
    """
    external_force = -lr * grad
    """
    return [], [], [-s_lr * v_grad for v_grad in v_grads]

def adadelta_force(options, s_lr, v_grads):
    """
    Adapted with modifications from arXiv:1212.5701
    """
    # Hyperparameters from Fractal are
    # rho = 0.99, e = 1e-20, clip = sqrt(1 / (1 - rho))
    rho  = options['force_ms_decay']
    e    = 1e-20
    clip = np.sqrt(1. / (1. - rho)).astype('float32') # 10.

    inits    = []
    updates  = []
    s_forces = []

    for v_grad in v_grads:
        # modded 0 init -> 1 init
        v_grad_ms = th.shared(np.ones_like(v_grad.get_value()))
        inits.append((v_grad_ms, tt.ones_like(v_grad_ms)))
        
        v_force_ms = th.shared(np.zeros_like(v_grad.get_value()))
        inits.append((v_force_ms, tt.zeros_like(v_force_ms)))

        s_new_grad_ms = rho * v_grad_ms + (1. - rho) * tt.sqr(v_grad)
        
        # modded s_lr**2, clip_elem
        s_force = -(tt.sqrt(v_force_ms + tt.sqr(s_lr))
                    * clip_elem(v_grad / (tt.sqrt(s_new_grad_ms) + e), clip))

        s_new_force_ms = rho * v_force_ms + (1. - rho) * tt.sqr(s_force)

        s_forces.append(s_force)
        updates.append((v_grad_ms , s_new_grad_ms ))
        updates.append((v_force_ms, s_new_force_ms))
    
    return inits, updates, s_forces

def rmsprop_force(options, s_lr, v_grads):
    """
    Adapted with modifications from
    http://www.cs.toronto.edu/~tijmen/csc321
          /slides/lecture_slides_lec6.pdf
    """
    # Hyperparameters from Fractal are
    # rho = 0.99, e = 1e-20, clip = sqrt(1 / (1 - rho))
    rho  = options['force_ms_decay']
    e    = 1e-20
    clip = np.sqrt(1. / (1. - rho)).astype('float32') # 10.

    inits    = []
    updates  = []
    s_forces = []

    for v_grad in v_grads:
        # modded 0 init -> 1 init
        v_grad_ms = th.shared(np.ones_like(v_grad.get_value()))
        inits.append((v_grad_ms, tt.ones_like(v_grad_ms)))

        s_new_grad_ms = rho * v_grad_ms + (1. - rho) * tt.sqr(v_grad)

        # modded clip_elem
        s_forces.append \
            (-s_lr * clip_elem(v_grad / (tt.sqrt(s_new_grad_ms) + e), clip))
        updates.append((v_grad_ms, s_new_grad_ms))
    
    return inits, updates, s_forces

def adam_force(options, s_lr, v_grads):
    """
    Adapted from arXiv:1412.6980
    """
    # Hyperparameters recommended in paper are
    # b1 = 0.9, b2 = 0.999, e = 1e-8
    b1 = options['force_adam_b1'] 
    b2 = options['force_adam_b2']
    e  = 1e-8

    inits    = [] 
    updates  = []
    s_forces = []

    v_t = th.shared(np.float32(0.)) # scalar
    inits.append((v_t, tt.zeros_like(v_t)))

    s_new_t = v_t + 1.
    s_calibrated_lr = s_lr * tt.sqrt(1. - b2**(s_new_t)) / (1. - b1**(s_new_t))

    for v_grad in v_grads:
        v_m = th.shared(np.zeros_like(v_grad.get_value()))
        inits.append((v_m, tt.zeros_like(v_m)))
        
        v_v = th.shared(np.zeros_like(v_grad.get_value()))
        inits.append((v_v, tt.zeros_like(v_v)))

        s_new_m = b1 * v_m + (1. - b1) * v_grad
        s_new_v = b2 * v_v + (1. - b2) * tt.sqr(v_grad)

        s_forces.append(-s_calibrated_lr * s_new_m / (tt.sqrt(s_new_v) + e))

        updates.append((v_m, s_new_m))
        updates.append((v_v, s_new_v))

    updates.append((v_t, s_new_t))

    return inits, updates, s_forces

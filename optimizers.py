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
Optimizers
- Implement the following signatures
    $_force (options, ones, s_lr, v_grads) \
        -> optim_f_inits, optim_f_updates, s_forces
    $_update(options, ones, s_forces) \
        -> optim_u_inits, optim_u_updates, s_increments
  where ones is a list of np.ones_like(param) in the same order as v_params
  (for obtaining shapes without eval()/get_value())
- v_grads, s_forces, & s_increments are also lists of same shapes & order

- To f_initialize_optimizer : optim_f_inits, optim_u_inits
- To f_update_v_params      : optim_f_updates, optim_u_updates, s_increments

- v_grads must be updated before applying updates returned here
- Order between tuples in the updates list doesn't matter
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import theano as th
import theano.tensor as tt
from utils import clip_elem

"""
Update functions to define increments for parameters, treating gradient as a
force along a potential hill with friction (unit mass & unit time step implied)
    v += - (1 - mu) v + gradient_force
where
    (1 - mu)      : dimensionless Stokes friction
    gradient_force: e.g., (-lr * grad) for vanilla_force
"""

def sgd_update(options, ones, s_forces):
    """
    Special case of momentum/nesterov with mu = 0
    (i.e., discrete version of critical damping)
        x += gradient_force
    """
    return [], [], s_forces

def momentum_update(options, ones, s_forces):
    """
    Naive update of momentum (velocity) with friction & gradient_force
        v = mu v + gradient_force
        x += v
    """
    mu = options['update_mu']

    inits = []
    updates = []
    s_increments = []

    for one, s_force in zip(ones, s_forces):
        v_v = th.shared(0. * one, name = 'momentum_v')
        inits.append((v_v, tt.zeros_like(v_v)))

        s_new_v = mu * v_v + s_force

        updates.append((v_v, s_new_v))
        s_increments.append(s_new_v)
    
    return inits, updates, s_increments

def nesterov_update(options, ones, s_forces):
    """
    Same as above, but with gradient_force at lookahead position
    (in terms of peeked-ahead params; arXiv:1212.0901 Eqs 6,7)
        x -= mu v
        v = mu v + gradient_force
        x += (1. + mu) v
    """
    mu = options['update_mu']

    inits = []
    updates = []
    s_increments = []

    for one, s_force in zip(ones, s_forces):
        v_v = th.shared(0. * one, name = 'nesterov_v')
        inits.append((v_v, tt.zeros_like(v_v)))

        s_new_v = mu * v_v + s_force

        updates.append((v_v, s_new_v))
        s_increments.append((1. + mu) * s_new_v - mu * v_v)

    return inits, updates, s_increments


"""
Force functions to define increment for velocity
(i.e., gradient_force excluding friction)
"""

def vanilla_force(options, ones, s_lr, v_grads):
    """
    gradient_force = -lr * grad
    """
    return [], [], [-s_lr * v_grad for v_grad in v_grads]

def adadelta_force(options, ones, s_lr, v_grads):
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

    for one, v_grad in zip(ones, v_grads):
        # modded 0 init -> 1 init
        v_grad_ms = th.shared(1. * one, name = 'adadelta_grad_ms')
        inits.append((v_grad_ms, tt.ones_like(v_grad_ms)))
        
        v_force_ms = th.shared(0. * one, name = 'adadelta_force_ms')
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

def rmsprop_force(options, ones, s_lr, v_grads):
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

    for one, v_grad in zip(ones, v_grads):
        # modded 0 init -> 1 init
        v_grad_ms = th.shared(1. * one, name = 'rmsprop_grad_ms')
        inits.append((v_grad_ms, tt.ones_like(v_grad_ms)))

        s_new_grad_ms = rho * v_grad_ms + (1. - rho) * tt.sqr(v_grad)

        # modded clip_elem
        s_forces.append \
            (-s_lr * clip_elem(v_grad / (tt.sqrt(s_new_grad_ms) + e), clip))
        updates.append((v_grad_ms, s_new_grad_ms))
    
    return inits, updates, s_forces

def adam_force(options, ones, s_lr, v_grads):
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

    v_t = th.shared(np.float32(0.), name = 'adam_t') # scalar
    inits.append((v_t, tt.zeros_like(v_t)))

    s_new_t = v_t + 1.
    s_calibrated_lr = s_lr * tt.sqrt(1. - b2**(s_new_t)) / (1. - b1**(s_new_t))

    for one, v_grad in zip(ones, v_grads):
        v_m = th.shared(0. * one, name = 'adam_m')
        inits.append((v_m, tt.zeros_like(v_m)))
        
        v_v = th.shared(0. * one, name = 'adam_v')
        inits.append((v_v, tt.zeros_like(v_v)))

        s_new_m = b1 * v_m + (1. - b1) * v_grad
        s_new_v = b2 * v_v + (1. - b2) * tt.sqr(v_grad)

        s_forces.append(-s_calibrated_lr * s_new_m / (tt.sqrt(s_new_v) + e))

        updates.append((v_m, s_new_m))
        updates.append((v_v, s_new_v))

    updates.append((v_t, s_new_t))

    return inits, updates, s_forces

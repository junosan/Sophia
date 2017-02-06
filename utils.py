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
Small helper functions
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import theano as th
import theano.tensor as tt
import string
import random

# Loss functions
# For display, average in tb dimensions (but not in i dimension)

def l2_loss(s_output_tbi, s_target_tbi):
    # D_err[loss] = err
    return tt.sum(tt.sqr(s_output_tbi - s_target_tbi) / 2.)

def l1_loss(s_output_tbi, s_target_tbi):
    # D_err[loss] = sgn(err)
    return tt.sum(tt.abs(s_output_tbi - s_target_tbi))

def huber_loss(s_output_tbi, s_target_tbi, delta):
    # D_err[loss] = clip_elem(err, delta)
    assert delta > 0.
    a = tt.abs(s_output_tbi - s_target_tbi)
    return tt.sum(tt.switch(a <= delta, tt.sqr(a) / 2.,
                                        delta * (a - delta / 2.)))


# Weight initializers
# Needs modification if used for ReLU or leaky ReLU nonlinearities:
#     http://lasagne.readthedocs.io/en/latest/modules/init.html

def unif_weight(options, n_in, n_out = None):
    """
    Uniform initalization from [-scale, scale)
    If n_out is None, assume 1D shape, otherwise 2D shape
    If matrix is square and init_use_ortho, use ortho_weight instead
    """
    if n_out is None:
        W = np.random.uniform(low  = -options['init_scale'],
                              high =  options['init_scale'],
                              size = n_in)
    elif n_in == n_out and options['init_use_ortho']:
        W = ortho_weight(n_in)
    else:
        W = np.random.uniform(low  = -options['init_scale'],
                              high =  options['init_scale'],
                              size = (n_in, n_out))
    return W.astype('float32')

def ortho_weight(dim):
    """
    Orthogonal weight init (i.e., all eigenvalues are of magnitude 1)
    """
    U = None
    while U is None:
        try:
            W = np.random.randn(dim, dim) # ~ Gaussian (mu = 0, sigma = 1)
            U, S, V = np.linalg.svd(W)
        except:
            pass # suppress probalistic failure
    return U.astype('float32')

def xavier_weight(n_in, n_out):
    """
    Xavier init
    """
    r = np.sqrt(6.) / np.sqrt(n_in + n_out)
    W = np.random.rand(n_in, n_out) * 2 * r - r
    # np.random.rand ~ Uniform [0, 1)
    # Gaussian is also possible -- see above reference
    return W.astype('float32')


# Weight clipping

def clip_norm(s_tensor, threshold):
    """
    Rescale given tensor to have norm at most equal to threshold
        threshold > 0.
    """
    assert threshold > 0.
    normsq = tt.sum(tt.sqr(s_tensor))
    return tt.switch(normsq > (threshold ** 2),
                     s_tensor / tt.sqrt(normsq) * threshold,
                     s_tensor)

def clip_elem(s_tensor, threshold):
    """
    Elementwise clipping to +-threshold
        threshold > 0.
    """
    assert threshold > 0.
    return tt.clip(s_tensor, -threshold, threshold)


# For avoiding name clash

def get_random_string(length = 8):
    return ''.join(random.SystemRandom().choice(string.ascii_uppercase
                                                + string.ascii_lowercase
                                                + string.digits) \
                   for _ in range(length))

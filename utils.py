#==========================================================================#
# Copyright (C) 2016 Hosang Yoon (hosangy@gmail.com) - All Rights Reserved #
# Unauthorized copying of this file, via any medium is strictly prohibited #
#                       Proprietary and confidential                       #
#==========================================================================#

"""
Small helper functions
"""

import numpy as np
import theano as th
import theano.tensor as tt

def MSE_loss(s_output_tbi, s_target_tbi):
    return tt.mean(tt.sqr(s_output_tbi - s_target_tbi))

# In the future, maybe add cross entropy loss, softmax activation, etc

def clip_norm(s_tensor, threshold):
    """
    Rescale given tensor to have norm at most equal to threshold
        threshold > 0.
    """
    assert threshold > 0.
    normsq = tt.sum(tt.sqr(s_tensor))
    return tt.switch(normsq > (threshold ** 2),
                     s_tensor / tt.sqrt(normsq) * threshold, s_tensor)

def clip_elem(s_tensor, threshold):
    """
    Elementwise clipping to +-threshold
        threshold > 0.
    """
    assert threshold > 0.
    return tt.minimum(threshold, tt.maximum(-threshold, s_tensor))

# Weight initializers
# Needs modification if used for ReLU or leaky ReLU nonlinearities:
#     http://lasagne.readthedocs.io/en/latest/modules/init.html

# scale = 0.1 is roughly sqrt(6)/sqrt(1024) (in line with Xavier init)
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
    # np.random.rand ~ Uniform [0, 1); Gaussian is also possible -- see above reference
    return W.astype('float32')


import string
import random

def get_random_string(length = 8):
    return ''.join(random.SystemRandom().choice( \
        string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(length))

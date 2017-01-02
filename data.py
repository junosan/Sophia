#==========================================================================#
# Copyright (C) 2016 Hosang Yoon (hosangy@gmail.com) - All Rights Reserved #
# Unauthorized copying of this file, via any medium is strictly prohibited #
#                       Proprietary and confidential                       #
#==========================================================================#

"""
Iterator class for time slices of shuffled sequence minibatches

- Creates time slices suitable for BPTT(h; h'), where
      h = window_size, h' = step_size (see doi:10.1162/neco.1990.2.4.490)
  For training, h = 2 h' is recommended
  For inference, h = h' is recommended (h > h' produces correct results,
                                        but wastes computation)
- For unroll_scan option to work in Net, window_size must be a compile time
  constant and hence cannot be changed once a Net is instantiated
- Sequences are assumed to be of arbitrary lengths, each with their own clocks
- Sequences are each given an id_idx to be used as an one-hot encoding index
  (may be ignored if not using ID information)
- Call discard_unfinished to use new sequences next iteration
- Use as
      for input_tbi, target_tbi, time_tb, id_idx_tb in data_iter:
          (loop content)
  where return values are np.ndarray's of dimensions
      input     float32     [window_size][batch_size][input_dim ]
      target    float32     [window_size][batch_size][target_dim]
      time      float32     [window_size][batch_size]
      id_idx    int32       [window_size][batch_size]
  randomly shuffled in 1-th (batch) dimension
- Time starts at 0. and increases by 1. each time index; 0. signals state reset
- Upon each iteration, 
    - Previous arrays are shifted by step_size to the left
    - New step_size amount of data are read from files and put on the right
    - Loss is to be calculated from the last step_size time indices
    - States are to be rewound to time index (step_size - 1) after propagation
- First few iterations may have zeros or irrelevant data on the left, but
  states will be reset when the real data starts and loss won't be calculated
  in or be propagated to the zero-padded/irrelevant region
- Unless stopped explicitly inside the loop, iterates indefinitely
"""

import numpy as np
from collections import OrderedDict

def seq_to_id(seq):
    return seq[seq.rfind('/') + 1 :].strip()

def build_id_idx(list_file):
    """
    This needs to be saved and used for inference as well
        id_idx['sequence id'] = one-hot encoding index
    """
    id_idx = OrderedDict()
    with open(list_file) as f:
        for seq in f:
            _id = seq_to_id(seq)
            if _id not in id_idx:
                id_idx[_id] = len(id_idx)
    return id_idx

class DataIter():
    def __iter__(self):
        return self

    def __init__(self, list_file, window_size, step_size,
                 batch_size, input_dim, target_dim, id_idx):
        self._data_root   = list_file[: list_file.rfind('/') + 1] # includes /
        self._window_size = window_size
        self.set_step_size(step_size)
        self._batch_size  = batch_size
        self._input_dim   = input_dim
        self._target_dim  = target_dim
        self._id_idx      = id_idx

        # buffers for last minibatch
        self._input_tbi  = np.zeros((window_size, batch_size, input_dim )) \
                             .astype('float32')
        self._target_tbi = np.zeros((window_size, batch_size, target_dim)) \
                             .astype('float32')
        self._time_tb    = np.zeros((window_size, batch_size)) \
                             .astype('float32')
        self._id_idx_tb  = np.zeros((window_size, batch_size)) \
                             .astype('int32')
        
        # buffers for currently open files
        # new files are read when time index reaches input.shape[0]
        self._t_idxs  = batch_size * [0] # time index cursors in files
        self._id_idxs = batch_size * [0] # id_idx values
        self._inputs  = batch_size * [np.zeros((0, input_dim )) \
                                        .astype('float32')]
        self._targets = batch_size * [np.zeros((0, target_dim)) \
                                        .astype('float32')]

        self._seqs = []
        with open(list_file) as f:
            for line in f:
                self._seqs.append(line.strip())
        self._n_seqs = len(self._seqs)
        assert self._n_seqs > 0
        
        self._shuffle()

    def _shuffle(self):
        self._seq_idx = 0
        self._seq_order = np.random.permutation(self._n_seqs) # of np.arange(n)

    def _pop_seq(self):
        if self._seq_idx >= self._n_seqs:
            self._shuffle()
        seq = self._seqs[self._seq_order[self._seq_idx]]
        self._seq_idx += 1
        return seq

    def _read(self, batch_idx):
        seq = self._pop_seq()

        self._t_idxs [batch_idx] = 0
        self._id_idxs[batch_idx] = self._id_idx[seq_to_id(seq)]

        # dtype '<f4' means little endian, float, 4 bytes
        self._inputs[batch_idx] = \
            np.fromfile(self._data_root + seq + '.input' , dtype = '<f4') \
              .reshape((-1, self._input_dim )).astype('float32')
        self._targets[batch_idx] = \
            np.fromfile(self._data_root + seq + '.target', dtype = '<f4') \
              .reshape((-1, self._target_dim)).astype('float32')
        assert self._inputs[batch_idx].shape[0] \
               == self._targets[batch_idx].shape[0]

    def discard_unfinished(self):
        for b in range(self._batch_size):
            if self._t_idxs[b] > 0:
                self._t_idxs[b] = self._inputs[b].shape[0]

    def set_step_size(self, step_size):
        assert self._window_size >= step_size
        self._step_size = step_size

    def next(self):
        def shift(arr, d): arr[: -d] = arr[d :]

        shift(self._input_tbi , self._step_size)
        shift(self._target_tbi, self._step_size)
        shift(self._time_tb   , self._step_size)
        shift(self._id_idx_tb , self._step_size)

        for b in range(self._batch_size):
            cur = self._window_size - self._step_size

            while cur < self._window_size:
                while self._t_idxs[b] >= self._inputs[b].shape[0]:
                    self._read(b)
                
                inc = min(self._window_size - cur,
                          self._inputs[b].shape[0] - self._t_idxs[b])
                rng_b = range(cur, cur + inc)
                rng_f = range(self._t_idxs[b], self._t_idxs[b] + inc)

                self._input_tbi [rng_b, b, :] = self._inputs [b][rng_f, :]
                self._target_tbi[rng_b, b, :] = self._targets[b][rng_f, :]
                self._time_tb   [rng_b, b]    = np.array(rng_f) \
                                                  .astype('float32')
                self._id_idx_tb [rng_b, b]    = self._id_idxs[b]

                cur += inc
                self._t_idxs[b] += inc
        
        return self._input_tbi, self._target_tbi, \
               self._time_tb, self._id_idx_tb

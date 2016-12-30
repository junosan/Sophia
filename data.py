#==========================================================================#
# Copyright (C) 2016 Hosang Yoon (hosangy@gmail.com) - All Rights Reserved #
# Unauthorized copying of this file, via any medium is strictly prohibited #
#                       Proprietary and confidential                       #
#==========================================================================#

"""
Iterator classes for shuffled sequence batches and their time slices
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

class DataBatchIter():
    """
    - for input_tbi, target_tbi, id_idx_b, batch_idx in DataBatchIter(args):
          (loop content)
    - Upon each iteration, returns tuple of np.ndarray's and batch_idx
          input     float32 [all_timesteps][batch_size][input_dim ]
          target    float32 [all_timesteps][batch_size][target_dim]
          id_idx    int     [batch_size]
          batch_idx (increases indefinitely every iteration)
      randomly shuffled in batch dimension
    - Files are read lazily when next() is called
    - Unless stopped explicitly inside loop, iterates indefinitely
    - Currently assumes that all sequences are of equal length
    """

    def __init__(self, list_file, input_dim, target_dim, batch_size, id_idx):
        self.data_root  = list_file[: list_file.rfind('/') + 1] # includes /
        self.input_dim  = input_dim
        self.target_dim = target_dim
        self.batch_size = batch_size
        self.id_idx     = id_idx

        self.seqs = []
        with open(list_file) as f:
            for line in f:
                self.seqs.append(line.strip())
        self.n_seqs = len(self.seqs)
        assert self.n_seqs > 0
        
        # '<f4' means little endian, float, 4 bytes
        #     https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
        input_ti = np.fromfile(self.data_root + self.seqs[0] + '.input',
                               dtype = '<f4').reshape((-1, input_dim ))
        
        # all files checked against this length (for now)
        self.all_timesteps = input_ti.shape[0]

        self.batch_idx = 0
        self.reset_seq_order()

    def __iter__(self):
        return self

    def reset_seq_order(self):
        self.seq_idx = 0
        self.seq_order = np.random.permutation(self.n_seqs) # of np.arange(n)
    
    def seq(self, seq_idx):
        return self.seqs[self.seq_order[seq_idx]]

    def read(self, seq_idx):
        """
        Returns tuple of np.ndarray's
            [all_timesteps][input_dim], [all_timesteps][target_dim]
        """
        input_ti  = np.fromfile(self.data_root + self.seq(seq_idx) + '.input' ,
                                dtype = '<f4').reshape((-1, self.input_dim ))
        target_ti = np.fromfile(self.data_root + self.seq(seq_idx) + '.target',
                                dtype = '<f4').reshape((-1, self.target_dim))

        assert input_ti .shape[0] == self.all_timesteps
        assert target_ti.shape[0] == self.all_timesteps

        return input_ti, target_ti

    def next(self):
        input_tbi  = np.zeros \
                     ((self.all_timesteps, self.batch_size, self.input_dim )) \
                     .astype('float32')
        target_tbi = np.zeros \
                     ((self.all_timesteps, self.batch_size, self.target_dim)) \
                     .astype('float32')
        id_idx_b   = np.zeros(self.batch_size).astype('int32')
        
        for b in range(self.batch_size):
            if self.seq_idx >= self.n_seqs:
                self.reset_seq_order()

            input_tbi[:, b, :], target_tbi[:, b, :] = self.read(self.seq_idx)
            id_idx_b[b] = self.id_idx[seq_to_id(self.seq(self.seq_idx))]

            self.seq_idx += 1
        
        self.batch_idx += 1

        return input_tbi, target_tbi, id_idx_b, self.batch_idx - 1


class TimeStepIter():
    """
    Creates time slices suitable for BPTT(h; h'), where
        h = window_size, h' = step_size (doi:10.1162/neco.1990.2.4.490)
    - window_size >= step_size; for inference, set window_size = step_size
    - Intended for use as inner loop of DataBatchIter's loop
    - Use as
          for input_step_tbi, target_step_tbi, time_t, \
                  last_tap, loss_tap in TimeStepIter(args):
              (loop content)
    - Upon each iteration, returns tuple of np.ndarray's and taps
          input     float32 [n_steps][batch_size][input_dim ]
          target    float32 [n_steps][batch_size][target_dim]
          time      float32 [n_steps]
          last_tap  int     scan should return as update (tt.switch)
                                last_tap == -1: prev_state
                                otherwise     : state[last_tap]
                            i.e., rewind states to last_tap after backprop
          loss_tap  int     loss should be calculated in [loss_tap : n_steps]
    - Time starts at 0. and increases by 1. each step
    - Assuming a sufficiently long sequence,
          First few: step_size <= n_steps <= window_size
                     (starts at step_size and increases by step_size
                      until window_size is reached)
                     n_loss_frames = step_size (from right)
          Middle   : n_steps = window_size
                     n_loss_frames = step_size (from right)
          Last     : window_size - step_size < n_steps <= window_size
                     0 < n_loss_frames <= step_size (from right)
    - For insufficiently long sequece, slightly different but still functional
    """
    
    def __init__(self, input_tbi, target_tbi, window_size, step_size):
        assert input_tbi.shape[0] == target_tbi.shape[0]
        assert window_size >= step_size

        self.input_tbi   = input_tbi
        self.target_tbi  = target_tbi
        self.step_size   = step_size

        self.all_timesteps = self.input_tbi.shape[0]
        self.time_idx = 0
        self.wind_idx = step_size - window_size

    def __iter__(self):
        return self
    
    def next(self):
        if self.time_idx >= self.all_timesteps:
            raise StopIteration()
        
        next_idx = min(self.time_idx + self.step_size, self.all_timesteps)
        
        shift = -min(0, self.wind_idx)
        rng = range(self.wind_idx + shift, next_idx)

        last_tap = min(max(0, self.step_size - shift) - 1, len(rng) - 1)
        loss_tap = self.time_idx - rng[0]

        self.wind_idx += self.step_size
        self.time_idx = next_idx

        return self.input_tbi[rng, :, :], \
               self.target_tbi[rng, :, :], \
               np.array(rng).astype('float32'), \
               last_tap, loss_tap

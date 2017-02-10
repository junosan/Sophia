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
Class for fwd propagating pre-trained nets in ensemble, one step at a time 
- Assume that the same Reshaper is used to process the data, but not
  necessarily on the same data (i.e., possibly different whitening matrix
  and/or different id_idx orders)
- Hence, receive and return data for all nets separately 
"""

from __future__ import absolute_import, division, print_function

import numpy as np
from net import Net
from collections import OrderedDict

class Ensemble():
    def __init__(self, workspaces, batch_size, indices):
        """
        Load pre-trained nets from files and prepare for fwd propagation
            workspaces  list    [workspace_0     , ..., workspace_(N-1)     ]
            batch_size  int     > 0
            indices     list    [batch_idx_list_0, ..., batch_idx_list_(N-1)]
        where
            batch_idx_list = [id_idx_0, ..., id_idx_(B-1)]
        is batch-dimension id_idx order in vec_in for run_one_step
        """
        self._n_nets = len(workspaces)
        self._batch_size = batch_size
        assert self._n_nets > 0 and batch_size > 0

        self._id_idx_nb = []
        assert len(indices) == self._n_nets
        for indice in indices:
            assert len(indice) == batch_size
            self._id_idx_nb.append(np.array(indice).astype('int32'))

        options = OrderedDict()
        options['step_size']  = 1 # can generalize to more than 1 if needed
        options['batch_size'] = batch_size

        self._input_dim = 0  # set below
        self._target_dim = 0 # set below
        self._nets = []
        self._props = [] # fwd propagators

        for workspace in workspaces:
            self._nets.append(Net(options, None, workspace))
            self._props.append(self._nets[-1].compile_f_fwd_propagate())

            if len(self._nets) == 1:
                self._input_dim, self._target_dim = self._nets[-1].dimensions()
            else:
                input_dim, target_dim = self._nets[-1].dimensions()
                assert self._input_dim  == input_dim and \
                       self._target_dim == target_dim

        self.reset()
    
    def reset(self):
        """
        Rewind to t = 0
        """
        self._time_t = np.zeros(1).astype('int32')
    
    def run_one_step(self, vec_in):
        """
        Inputs:
            vec_in  np.ndarray  [n_nets][batch_size][input_dim]  (flattened)
        Returns:
            vec_out np.ndarray  [n_nets][batch_size][target_dim] (flattened)
        """
        input_nbi  = vec_in.astype('float32').reshape \
                         ((self._n_nets, self._batch_size, self._input_dim ))
        output_nbi = np.zeros \
                         ((self._n_nets, self._batch_size, self._target_dim)) \
                         .astype('float32')

        for i, f in enumerate(self._props):
            # f(input_tbi, time_t, id_idx_tb) -> [output_tbi]
            output_nbi[i] = f(input_nbi[i][None, :, :], self._time_t,
                              self._id_idx_nb[i][None, :])[0] # _bi = _1bi 

        self._time_t[0] += 1
        return output_nbi.reshape(-1)

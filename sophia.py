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
Program for real-time inference via ZeroMQ IPC/TCP with an external process
"""

from __future__ import absolute_import, division, print_function

import zmq
import numpy as np
from ensemble import Ensemble

def main():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    
    # using IPC here, but also supports TCP if communicating over a network
    socket.bind("ipc:///tmp/sophia_ipc")

    # parse first msg:
    #     'workspace_0;...;workspace_(N-1);batch_size
    #      idx_0;...;idx_(B-1)  (for workspace 0)
    #      ...
    #      idx_0;...;idx_(B-1)' (for workspace N-1)
    # which declares paths for N models and batch_size (B) number of streams
    #
    # idx_i for stream i is the 0-based index of ID corresponding to stream i
    # in ids.order (saved during training); idx_i can be set
    # to an arbitrary number if not using options['learn_id_embedding']
    lines = str(socket.recv()).splitlines()
    assert len(lines) > 0

    workspaces = lines[0][: lines[0].rfind(';')].split(';')
    batch_size = int(lines[0][lines[0].rfind(';') + 1 :])
    assert len(lines) == len(workspaces) + 1

    indices = [] # list of lists
    for line in lines[1 :]:
        indice = [int(idx) for idx in line.split(';')]
        assert len(indice) == batch_size
        indices.append(indice)
    
    ensemble = Ensemble(workspaces, batch_size, indices) # time consuming
    socket.send('ready') # to fulfill REQ/REP pattern

    while True:
        """
        Receives and returns raw binary buffer (little endian float32)
            in : [n_nets][batch_size][input_dim ] (flattened to 1-dim)
            out: [n_nets][batch_size][target_dim] (flattened to 1-dim)
        """
        inp = np.frombuffer(socket.recv(), dtype = '<f4')

        # signal end by sending one std::nanf("")
        if np.isnan(inp[0]):
            break

        socket.send(ensemble.run_one_step(inp))

if __name__ == '__main__':
    main()

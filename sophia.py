#==========================================================================#
# Copyright (C) 2016 Hosang Yoon (hosangy@gmail.com) - All Rights Reserved #
# Unauthorized copying of this file, via any medium is strictly prohibited #
#                       Proprietary and confidential                       #
#==========================================================================#

"""
Script for real time inference via ZeroMQ IPC with Sibyl
"""

import zmq
import numpy as np
from ensemble import Ensemble

def main():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("ipc:///tmp/sophia_sibyl")

    # First msg:
    #     'workspace_0;...;workspace_(N-1);batch_size
    #      idx_0;...;idx_(B-1)  (for workspace 0)
    #      ...
    #      idx_0;...;idx_(B-1)' (for workspace N-1)
    lines = str(socket.recv()).splitlines()
    assert len(lines) > 0

    workspaces = lines[0][: lines[0].rfind(';')].split(';')
    batch_size = int(lines[0][lines[0].rfind(';') + 1 :])
    assert len(lines) == len(workspaces) + 1

    indices = []
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
        msg = socket.recv()
        inp = np.frombuffer(msg, dtype = '<f4')

        # Sibyl signals end by sending one std::nanf("")
        if np.isnan(inp[0]):
            break

        socket.send(ensemble.run_one_step(inp))

if __name__ == '__main__':
    main()

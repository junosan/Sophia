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

    # First msg: 'workspace_0;...;workspace_(N-1);batch_size'
    beg = str(socket.recv())
    ensemble = Ensemble(beg[: beg.rfind(';')].split(';'), \
                        int(beg[beg.rfind(';') + 1 :]))
    socket.send('ready')

    while True:
        """
        Receives and returns raw binary buffer (little endian float32)
            in : [n_nets][batch_size][input_dim]  (flattened to 1-dim)
            out: [n_nets][batch_size][target_dim] (flattened to 1-dim)
        """
        msg = socket.recv()
        input = np.frombuffer(msg, dtype = '<f4')

        # Sibyl signals end by sending one std::nanf("")
        if np.isnan(input[0]):
            break

        socket.send(ensemble.run_one_step(input))

if __name__ == '__main__':
    main()

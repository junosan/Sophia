#==========================================================================#
# Copyright (C) 2016 Hosang Yoon (hosangy@gmail.com) - All Rights Reserved #
# Unauthorized copying of this file, via any medium is strictly prohibited #
#                       Proprietary and confidential                       #
#==========================================================================#

"""
Script for training

Use as:
    THEANO_FLAGS=floatX=float32,device=$DEV,lib.cnmem=1 python -u train.py \
        --data_dir=$DATA_DIR --save_to=$WORKSPACE_DIR/workspace_$NAME \
        [--load_from=$WORKSPACE_DIR/workspace_$LOAD] \
        | tee -a $WORKSPACE_DIR/$NAME".log"
    where $DEV=gpu0, etc (it is ok to set $NAME == $LOAD)
"""

from __future__ import print_function # for end option in print()
from collections import OrderedDict
import argparse
from net import Net
from data import build_id_idx, DataIter
import time
import numpy as np
from subprocess import call
import sys

def main():
    options = OrderedDict()

    options['input_dim']          = 44
    options['target_dim']         = 1
    options['unit_type']          = 'LSTM'     # FC/LSTM/GRU
    options['lstm_peephole']      = True
    options['net_width']          = 1024
    options['net_depth']          = 4
    options['batch_size']         = 64
    options['window_size']        = 128
    options['step_size']          = 64
    options['init_scale']         = 0.02
    options['init_use_ortho']     = False
    options['learn_init_states']  = True
    options['layer_norm']         = False
    options['skip_connection']    = False
    options['learn_clock_params'] = False
    # options['clock_t_exp_lo']     = 1.         # for learn_clock_params
    # options['clock_t_exp_hi']     = 6.         # for learn_clock_params
    # options['clock_r_on']         = 0.2        # for learn_clock_params
    # options['clock_leak_rate']    = 0.001      # for learn_clock_params
    options['id_count']           = 199        # do not comment out
    options['learn_id_embedding'] = False
    # options['id_embedding_dim']   = 16
    # options['grad_norm_clip']     = 2.         # comment out to turn off
    options['update_type']        = 'nesterov' # sgd/momentum/nesterov
    options['update_mu']          = 0.9        # for momentum/nesterov
    options['force_type']         = 'adadelta' # vanilla/adadelta/rmsprop/adam
    options['force_ms_decay']     = 0.99       # for adadelta/rmsprop
    # options['force_adam_b1']      = 0.9
    # options['force_adam_b2']      = 0.999
    options['frames_per_epoch']   = 8 * 1024 * 1024
    options['lr_init_val']        = 1e-5
    options['lr_lower_bound']     = 1e-7
    options['lr_decay_rate']      = 0.5
    options['max_retry']          = 10
    options['unroll_scan']        = True      # for faster training time
                                              # at the expense of compile time

    if options['unroll_scan']:
        # value 32 found empirically; may need change with different options
        sys.setrecursionlimit(32 * options['window_size'])

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir' , type = str, required = True)
    parser.add_argument('--save_to'  , type = str, required = True)
    parser.add_argument('--load_from', type = str)    
    args = parser.parse_args()

    assert 0 == call(str('mkdir -p ' + args.save_to).split())
    assert 0 == call(str('cp ' + args.data_dir + '/mean.matrix '
                         + args.save_to).split())
    assert 0 == call(str('cp ' + args.data_dir + '/whitening.matrix '
                         + args.save_to).split())

    def print_hline(): print(''.join('-' for _ in range(70)))
    lapse_from = lambda start: ('(' + ('%.1f' % (time.time() - start)).rjust(6)
                                + ' sec)')

    print_hline() # -----------------------------------------------------------
    print('Data location: ' + args.data_dir)
    if args.load_from is not None:
        print('Re-train from: ' + args.load_from)
    print('Save model to: ' + args.save_to)

    # save internal ID order
    id_idx = build_id_idx(args.data_dir + '/train.list')
    if len(id_idx) != options['id_count']:
        print('[Error] Unique ID count ' + str(len(id_idx))
              + ' != options[\'id_count\'] ' + str(options['id_count']))
        assert False
    with open(args.save_to + '/ids.order', 'w') as f:
        f.write(';'.join(id_idx.iterkeys())) # code_0;...;code_N-1

    print_hline() # -----------------------------------------------------------
    print('Options:')
    maxlen = max([len(v) for v in options.keys()])
    for k, v in options.iteritems():
        print('    ' + k.ljust(maxlen) + ' : ' + str(v))

    print_hline() # -----------------------------------------------------------
    print('Setting up expression graph...   ', end = ' ')
    start = time.time()
    net = Net(options, args.save_to, args.load_from)
    print(lapse_from(start)) 

    print('Compiling fwd/bwd propagators... ', end = ' ')
    start = time.time()
    f_fwd_bwd_propagate = net.compile_f_fwd_bwd_propagate()
    f_fwd_propagate     = net.compile_f_fwd_propagate()
    print(lapse_from(start))

    print('Compiling updater/initializers...', end = ' ')
    start = time.time()
    f_update_v_params = net.compile_f_update_v_params()
    f_initialize_optimizer = net.compile_f_initialize_optimizer()
    print(lapse_from(start))

    # NOTE: window_size must be the same as that given to Net
    train_data = DataIter(list_file   = args.data_dir + '/train.list',
                          window_size = options['window_size'],
                          step_size   = options['step_size'],
                          batch_size  = options['batch_size'],
                          input_dim   = options['input_dim'],
                          target_dim  = options['target_dim'],
                          id_idx      = id_idx)
    dev_data   = DataIter(list_file  = args.data_dir + '/dev.list',
                          window_size = options['window_size'],
                          step_size   = options['step_size'],
                          batch_size  = options['batch_size'],
                          input_dim   = options['input_dim'],
                          target_dim  = options['target_dim'],
                          id_idx      = id_idx)
    
    chunk_size = options['step_size'] * options['batch_size']
    trained_frames_per_epoch = \
        (options['frames_per_epoch'] // chunk_size) * chunk_size

    def run_epoch(data_iter, lr_cur):
        """
        lr_cur sets the running mode
            float   training
            None    inference
        """
        is_training = lr_cur is not None
        if is_training: # apply BPTT(window_size; step_size)
            step_size = options['step_size']
        else: # set next_prev_idx = window_size - 1 for efficiency
            step_size = options['window_size']
        frames_per_step = step_size * options['batch_size']

        data_iter.discard_unfinished()
        data_iter.set_step_size(step_size)

        loss_sum = 0.
        frames_seen = 0

        for input_tbi, target_tbi, time_tb, id_idx_tb in data_iter:
            if is_training:
                loss = f_fwd_bwd_propagate(input_tbi, target_tbi, 
                                           time_tb, id_idx_tb, step_size)
            else:
                loss = f_fwd_propagate(input_tbi, target_tbi, 
                                       time_tb, id_idx_tb, step_size)
            
            loss_sum    += np.asscalar(loss[0])
            frames_seen += frames_per_step
            
            if is_training:
                f_update_v_params(lr_cur)
            
            if frames_seen >= trained_frames_per_epoch:
                break
        return np.float32(loss_sum / frames_seen)
    

    """
    Scheduled learning rate annealing with patience
    From: https://github.com/KyuyeonHwang/Fractal
    """

    # Names for saving/loading
    name_pivot = '0'
    name_prev  = '1'
    name_best  = None # auto

    total_trained_frames = 0
    total_trained_frames_at_best = 0
    total_trained_frames_at_pivot = 0
    total_discarded_frames = 0

    loss_pivot = 0.
    loss_prev  = 0.
    loss_best  = 0.

    cur_retry = 0

    net.save_v_params_to_workspace(name_prev)
    net.save_v_params_to_workspace(name_best)

    lr = options['lr_init_val']
    f_initialize_optimizer()

    while True:
        print_hline() # -------------------------------------------------------
        print('Training...  ', end = ' ')
        start = time.time()
        _ = run_epoch(train_data, lr)
        print(lapse_from(start))

        total_trained_frames += trained_frames_per_epoch

        print('Evaluating...', end = ' ')
        start = time.time()
        loss_cur = run_epoch(dev_data, None)
        print(lapse_from(start))

        print('Total trained frames  : '
              + str(total_trained_frames  ).rjust(12))
        print('Total discarded frames: '
              + str(total_discarded_frames).rjust(12))

        print('Loss: ' + str(loss_cur), end = ' ')

        if np.isnan(loss_cur):
            loss_cur = np.float32('inf')
        
        if total_trained_frames == trained_frames_per_epoch or \
               loss_cur < loss_best:
            print('(best)', end = '')

            loss_best = loss_cur
            total_trained_frames_at_best = total_trained_frames
            net.save_v_params_to_workspace(name_best)
        print('')

        if total_trained_frames > trained_frames_per_epoch and \
               loss_prev < loss_cur:
            print_hline() # ---------------------------------------------------
            
            cur_retry += 1
            if cur_retry > options['max_retry']:
                cur_retry = 0

                lr *= options['lr_decay_rate']

                if lr < options['lr_lower_bound']:
                    break
                
                net.load_v_params_from_workspace(name_pivot)
                net.save_v_params_to_workspace(name_prev)

                discarded_frames \
                    = total_trained_frames - total_trained_frames_at_pivot
                
                print('Discard recently trained '
                      + str(discarded_frames) + ' frames')
                print('New learning rate: ' + str(lr))
                
                f_initialize_optimizer()

                total_discarded_frames += discarded_frames
                total_trained_frames = total_trained_frames_at_pivot
                loss_prev = loss_pivot
            else:
                print('Retry count: ' + str(cur_retry)
                      + ' / ' + str(options['max_retry']))
        else:
            cur_retry = 0

            # prev goes to pivot & cur goes to prev
            loss_pivot, loss_prev = loss_prev, loss_cur
            name_pivot, name_prev = name_prev, name_pivot

            net.save_v_params_to_workspace(name_prev)

            total_trained_frames_at_pivot \
                = total_trained_frames - trained_frames_per_epoch
    
    net.load_v_params_from_workspace(name_best)
    net.remove_params_file_from_workspace(name_pivot)
    net.remove_params_file_from_workspace(name_prev)

    total_discarded_frames \
        += total_trained_frames - total_trained_frames_at_best
    total_trained_frames = total_trained_frames_at_best

    print('')
    print('Done')
    print('Total trained frames  : ' + str( total_trained_frames ).rjust(12))
    print('Total discarded frames: ' + str(total_discarded_frames).rjust(12))
    print('')

    print('Best network:')
    print('Train set')
    loss_train = run_epoch(train_data, None)
    print('Loss: ' + str(loss_train))
    print('Dev set')
    loss_dev = run_epoch(dev_data, None)
    print('Loss: ' + str(loss_dev))

if __name__ == '__main__':
    main()

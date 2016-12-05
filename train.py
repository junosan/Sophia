#==========================================================================#
# Copyright (C) 2016 Hosang Yoon (hosangy@gmail.com) - All Rights Reserved #
# Unauthorized copying of this file, via any medium is strictly prohibited #
#                       Proprietary and confidential                       #
#==========================================================================#

"""
Script for training
Use as:
    THEANO_FLAGS=floatX=float32,device=$DEV,lib.cnmem=1 python -u train.py \
        --data_dir=$DATA_DIR --save_to=$WORKSPACE_DIR/workspace_$NAME0 \
        [--load_from=$WORKSPACE_DIR/workspace_$NAME1]
    where $DEV=gpu0, etc
"""

from __future__ import print_function # switches print to Python 3 version for end option
from collections import OrderedDict
import argparse
from net import Net
from data_iterators import DataBatchIter, TimeStepIter
import time
import numpy as np

def main():
    options = OrderedDict()

    options['input_dim']          = 44
    options['target_dim']         = 1
    options['unit_type']          = 'LSTM'
    options['net_width']          = 1024
    options['net_depth']          = 4
    options['batch_size']         = 64
    options['step_size']          = 128
    options['rolling_first_step'] = True
    options['learn_init_states']  = False
    options['layer_norm']         = False
    options['learn_clock_params'] = False
    options['clock_t_exp_lo']     = 1.
    options['clock_t_exp_hi']     = 7.
    options['clock_r_on']         = 0.1
    options['clock_leak_rate']    = 0.001
    options['grad_clip']          = 2.
    options['optimizer']          = 'adam'
    options['lr_init_val']        = 0.001
    options['lr_lower_bound']     = 0.001
    options['lr_decay_rate']      = 0.5
    options['max_retry']          = 2
    options['frames_per_epoch']   = 8 * 1024 * 1024


    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir' , type = str, required = True)
    parser.add_argument('--save_to'  , type = str, required = True)
    parser.add_argument('--load_from', type = str)    
    args = parser.parse_args()

    print('----------------------------------------------------------------------')
    print('Data location: ' + args.data_dir)
    if args.load_from is not None:
        print('Re-train from: ' + args.load_from)
    print('Save model to: ' + args.save_to)
    print('----------------------------------------------------------------------')
    print('Options:')
    maxlen = max([len(v) for v in options.keys()])
    for k, v in options.iteritems():
        print('    ' + k.ljust(maxlen) + ' : ' + str(v))
    print('----------------------------------------------------------------------')

    lapse_from = lambda start: '(' + ('%.1f' % (time.time() - start)).rjust(6) + ' sec)' 
    
    # from datetime import datetime
    # datetime.now().strftime('%Y%m%d %H:%M:%S')

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
    f_update_v_params      = net.compile_f_update_v_params()
    f_initialize_states    = net.compile_f_initialize_states()
    f_initialize_optimizer = net.compile_f_initialize_optimizer()
    print(lapse_from(start))

    train_data = DataBatchIter(list_file  = args.data_dir + '/train.list',
                               input_dim  = options['input_dim'],
                               target_dim = options['target_dim'],
                               batch_size = options['batch_size'])
    dev_data   = DataBatchIter(list_file  = args.data_dir + '/dev.list',
                               input_dim  = options['input_dim'],
                               target_dim = options['target_dim'],
                               batch_size = options['batch_size'])
    
    def run_epoch(bool_train, data_iter, lr, frames_per_epoch, options, 
                  f_initialize_states, f_propagate, f_update_v_params):
        losses = []
        frames = []
        frames_seen = 0
        finished = False
        for input_tbi, target_tbi, batch_idx in data_iter:
            f_initialize_states()
            for input_step_tbi, target_step_tbi, time_t in \
                    TimeStepIter(input_tbi       = input_tbi,
                                 target_tbi      = target_tbi,
                                 step_size       = options['step_size'],
                                 first_step_hint = batch_idx if options['rolling_first_step'] \
                                                   else 0):
                losses.append(f_propagate(input_step_tbi, target_step_tbi, time_t))
                frame = input_step_tbi.shape[0] * input_step_tbi.shape[1]
                frames.append(frame)
                frames_seen += frame
                if bool_train:
                    f_update_v_params(lr)
                if frames_seen >= frames_per_epoch:
                    finished = True
                    break
            if finished:
                break
        return np.sum(np.array(losses) * np.array(frames)) / float(frames_seen), frames_seen
    
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

    is_first = True
    actual_frames_per_epoch = options['frames_per_epoch']

    while True:
        print('----------------------------------------------------------------------')
        print('Training...  ', end = ' ')
        start = time.time()
        _, trained_frames = run_epoch(True, train_data, lr, options['frames_per_epoch'], options,
                                      f_initialize_states, f_fwd_bwd_propagate, f_update_v_params)
        print(lapse_from(start))

        total_trained_frames += trained_frames
        if is_first:
            actual_frames_per_epoch = trained_frames # as all sequences are forced to be of equal
            is_first = False                         # lengths currently, this will stay constant

        print('Evaluating...', end = ' ')
        start = time.time()
        loss_cur, _ = run_epoch(False, dev_data, None, options['frames_per_epoch'], options,
                                f_initialize_states, f_fwd_propagate, None)
        print(lapse_from(start))

        print('Total  trained  frames: ' + str(total_trained_frames  ).rjust(12))
        print('Total discarded frames: ' + str(total_discarded_frames).rjust(12))

        print('Loss: ' + str(loss_cur), end = ' ')

        if np.isnan(loss_cur):
            loss_cur = float('Inf')
        
        if total_trained_frames == actual_frames_per_epoch or loss_cur < loss_best:
            print('(best)', end = '')

            loss_best = loss_cur
            total_trained_frames_at_best = total_trained_frames
            net.save_v_params_to_workspace(name_best)
        print('')

        if total_trained_frames > actual_frames_per_epoch and loss_prev < loss_cur:
            print('----------------------------------------------------------------------')
            
            cur_retry += 1
            if cur_retry > options['max_retry']:
                cur_retry = 0

                lr *= options['lr_decay_rate']

                if lr < options['lr_lower_bound']:
                    break
                
                net.load_v_params_from_workspace(name_pivot)
                net.save_v_params_to_workspace(name_prev)

                discarded_frames = total_trained_frames - total_trained_frames_at_pivot
                print('Discard recently trained ' + str(discarded_frames) + ' frames')
                print('New learning rate: ' + str(lr))
                
                f_initialize_optimizer()

                total_discarded_frames += discarded_frames
                total_trained_frames = total_trained_frames_at_pivot
                loss_prev = loss_pivot
            else:
                print('Retry count: ' + str(cur_retry) + ' / ' + str(options['max_retry']))
        else:
            cur_retry = 0

            # prev goes to pivot & cur goes to prev
            loss_pivot = loss_prev
            loss_prev  = loss_cur

            temp = name_pivot
            name_pivot = name_prev
            name_prev = temp

            net.save_v_params_to_workspace(name_prev)

            total_trained_frames_at_pivot = total_trained_frames - actual_frames_per_epoch
    
    net.load_v_params_from_workspace(name_best)
    net.remove_params_file_from_workspace(name_pivot)
    net.remove_params_file_from_workspace(name_prev)

    total_discarded_frames += total_trained_frames - total_trained_frames_at_best
    total_trained_frames = total_trained_frames_at_best

    print('')
    print('Done')
    print('Total  trained  frames: ' + str( total_trained_frames ).rjust(12))
    print('Total discarded frames: ' + str(total_discarded_frames).rjust(12))
    print('')

    print('Best network:')
    print('Train set')
    loss_train, _ = run_epoch(False, train_data, None, options['frames_per_epoch'], options,
                              f_initialize_states, f_fwd_propagate, None)
    print('    Loss: ' + str(loss_train))
    print('Dev set')
    loss_dev, _   = run_epoch(False, dev_data, None, options['frames_per_epoch'], options,
                              f_initialize_states, f_fwd_propagate, None)
    print('    Loss: ' + str(loss_dev))

if __name__ == '__main__':
    main()


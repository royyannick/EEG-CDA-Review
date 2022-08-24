from operator import xor
import numpy as np
import pandas as pd

import mne
from mne import Epochs, pick_channels
from autoreject import AutoReject

#==================================================================
# General Functions
#==================================================================
def get_key(d, val):
    for k in d.keys():
        for t in d[k]:
            if t == val:
                return k

    return None


def get_specific_events(events, conds, sides, perfs, triggers, internal_triggers):
    triggers_cond = []
    triggers_side = []
    triggers_perf = []

    specific_events = dict()
    for cur_cond in conds:
        specific_events[cur_cond] = dict()
        for t in triggers[cur_cond]: triggers_cond.append(t)

        for cur_side in sides:
            specific_events[cur_cond][cur_side] = dict()
            for t in triggers[cur_side]: 
                triggers_side.append(t)

            for cur_perf in perfs:
                specific_events[cur_cond][cur_side][cur_perf] = np.array([])
                for t in triggers[cur_perf]: triggers_perf.append(t)

    # Find Events from File.
    total_count = 0
    total_skipped = 0
    for i, e in enumerate(events):
        if e[2] in triggers_cond: # Starting from condition. Will go forward for result.
            cur_cond = get_key(triggers, e[2])

            # Find next perf trigger to classify this trial based on side.
            j=i+1
            cur_side = None
            while (cur_side is None) and (j < len(events)):
                if events[j, 2] in triggers_side:
                    cur_side = get_key(triggers, events[j, 2])                    
                else:
                    if events[j, 2] in triggers_cond:
                        #raise ValueError('Overlapping Events with no Accuracy/Perf!')
                        print('Overlapping Events with no Accuracy/Perf! Skipping... {} at {}'.format(events[j, 2], events[j, 0]))
                        total_skipped = total_skipped + 1
                    j=j+1

            # All good from EEG file. (to confirm!)
            cur_perf = 'good'

            if cur_side is not None and cur_perf is not None:
                cur_event = events[j].copy()
                cur_event[2] = internal_triggers['{}-{}-{}'.format(cur_cond,cur_side,cur_perf)] # Modify the value to make it possible to separate them later.
                #print('{}. Adding: {} - {} - {}'.format(total_count, cur_cond, cur_side, cur_perf))
                specific_events[cur_cond][cur_side][cur_perf] = np.vstack((specific_events[cur_cond][cur_side][cur_perf], cur_event)) if len(specific_events[cur_cond][cur_side][cur_perf]) else cur_event
                total_count = total_count + 1


    for cur_cond in specific_events.keys():
        for cur_side in specific_events[cur_cond].keys():
            for cur_perf in specific_events[cur_cond][cur_side].keys():
                if (len(specific_events[cur_cond][cur_side][cur_perf].shape) == 1) and (specific_events[cur_cond][cur_side][cur_perf].shape[0] == 3):
                    specific_events[cur_cond][cur_side][cur_perf] = specific_events[cur_cond][cur_side][cur_perf].reshape((1,3))

    print("Total: {} ({} Skipped do to overlapping events with missing triggers.)".format(total_count, total_skipped))
    
    return specific_events
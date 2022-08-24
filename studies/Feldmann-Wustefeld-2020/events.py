from operator import xor
import numpy as np
import pandas as pd
from mne import Epochs, pick_channels
from scipy.io import loadmat

FELDMANN2020_PATH_BEHAVIOR = 'D:\\CLNT\\Data\\Open Datasets\\Feldmann-Westefel 2020\\rawBehavior\\'
#FELDMANN2020_PATH_BEHAVIOR = '/Users/nick/Documents/PhD/CDA Datasets/Feldmann-Westefel 2020/rawBehavior/'

#==================================================================
# General Functions
#==================================================================
def get_key(d, val):
    for k in d.keys():
        for t in d[k]:
            if t == val:
                return k

    return None

def get_specific_events(events, conds, sides, perfs, triggers, subject_id, internal_triggers):
    behavfilepath = FELDMANN2020_PATH_BEHAVIOR + '{}_Kamo_main.mat'.format(subject_id)
    behav = loadmat(behavfilepath)

    acc = behav['data']['accuracy'][0][0]
    acc = acc[~np.isnan(acc)]

    # -----
    trigger_clean_trials = 33
    
    triggers_cond = []
    triggers_side = []
    triggers_perf = [] 

    specific_events = dict()
    for cur_cond in conds:
        specific_events[cur_cond] = dict()
        for t in triggers[cur_cond]: triggers_cond.append(t)

        for cur_side in sides:
            specific_events[cur_cond][cur_side] = dict()
            for t in triggers[cur_side]: triggers_side.append(t)

            # Behavior File for Perf. Not in Triggers!
            for cur_perf in perfs:
                specific_events[cur_cond][cur_side][cur_perf] = np.array([])
            #    for t in triggers[cur_perf]: triggers_perf.append(t)

    total_count = 0
    total_skipped = 0
    errorTriggers = 0
    nbtrials = 0
    for i, e in enumerate(events):
        errorTriggers = 0
        # If event is a clean one.
        if(e[2] == trigger_clean_trials):
            j = i

            # Find the previous events which was a condition trigger.
            while((events[j, 2] not in triggers['all']) and (j >= 0) and (errorTriggers == 0)): # should wrap in a try/catch
                j = j -1

                if(events[j, 2] == trigger_clean_trials):
                    print('Overlapping Events with no Cond! Skipping...')
                    total_skipped = total_skipped + 1
                    errorTriggers = 1

            # Check if we are skipping (i.e. not in cond we care about)
            if (errorTriggers == 0) and (j >= 0):
                cur_cond = None
                for cond in conds:
                    if events[j, 2] in triggers[cond]:
                        cur_cond = cond
                        #print("\t{} -> {}".format(events[j, 2], cond))

                cur_side = None
                for side in sides:
                    if events[j, 2] in triggers[side]:
                        cur_side = side

                if(int(acc[nbtrials])):
                    cur_perf = 'good'
                else:
                    cur_perf = 'bad'

                if cur_side is not None and cur_perf is not None:
                    cur_event = events[j].copy()
                    cur_event[2] = internal_triggers['{}-{}-{}'.format(cur_cond,cur_side,cur_perf)] # Modify the value to make it possible to separate them later.
                    #print('{}. Adding: {} - {} - {}'.format(total_count, cur_cond, cur_side, cur_perf))
                    specific_events[cur_cond][cur_side][cur_perf] = np.vstack((specific_events[cur_cond][cur_side][cur_perf], cur_event)) if len(specific_events[cur_cond][cur_side][cur_perf]) else cur_event
                    total_count = total_count + 1

                nbtrials += 1

    print("Nb Trials: {} | Total Added: {} | Total Skipped: {} ".format(nbtrials, total_count, total_skipped))

    for cur_cond in specific_events.keys():
        for cur_side in specific_events[cur_cond].keys():
            for cur_perf in specific_events[cur_cond][cur_side].keys():
                if (len(specific_events[cur_cond][cur_side][cur_perf].shape) == 1) and (specific_events[cur_cond][cur_side][cur_perf].shape[0] == 3):
                    specific_events[cur_cond][cur_side][cur_perf] = specific_events[cur_cond][cur_side][cur_perf].reshape((1,3))

    return specific_events
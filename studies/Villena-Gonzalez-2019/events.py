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

    # Prepare structure to contain the events (from MNE) based on conds/sides/perfs and triggers.
    specific_events = dict()
    for cur_cond in conds:
        specific_events[cur_cond] = dict()
        for t in triggers[cur_cond]: triggers_cond.append(t)

        for cur_side in sides:
            specific_events[cur_cond][cur_side] = dict()
            for t in triggers[cur_side]: triggers_side.append(t)

            for cur_perf in perfs:
                specific_events[cur_cond][cur_side][cur_perf] = np.array([])
                for t in triggers[cur_perf]: triggers_perf.append(t)

    # Find Events from File.
    total_count = 0
    total_skipped = 0
    for i, e in enumerate(events):
        if e[2] in triggers_cond: # Starting from condition. Will go forward for result.
            cur_cond = get_key(triggers, e[2])

            # Find next perf trigger to classify this trial based on perf.
            j=i+1
            cur_perf = None
            while (cur_perf is None) and (j < len(events)):
                if events[j, 2] in triggers_perf:
                    cur_perf = get_key(triggers, events[j, 2])                        
                else:
                    if events[j, 2] in triggers_cond:
                        #raise ValueError('Overlapping Events with no Accuracy/Perf!')
                        print('Overlapping Events with no Accuracy/Perf! Skipping...')
                        total_skipped = total_skipped + 1
                    j=j+1

            # Look at event trigger to classify side. (conds and sides are overlapping triggers)
            cur_side = None
            if e[2] in triggers_side: # Should always be true.
                for side in sides:
                    if e[2] in triggers[side]:
                        cur_side = side
            else:
                #raise ValueError('Overlapping Events with no Side! Skipping...')
                print('Should not happen! Skipping...')
                total_skipped = total_skipped + 1

            if cur_side is not None and cur_perf is not None:
                cur_event = e.copy()
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


def get_epochs(eeg, specific_events, epoch_length, epoch_tmin, baseline_corr, clean=False):
    print("====================== EPOCHING ======================")
    print("tmin:{}, tmax:{}, baseline={}".format(epoch_tmin, epoch_length, baseline_corr))
    
    # 1- Combine all relevant events.
    # 2- Extract all Epochs, to correct baseline the same way.
    # 3- Clean Epochs (if clean True)
    # 4- Separate Epochs.

    all_events = []
    for cur_cond in specific_events.keys():
        for cur_side in specific_events[cur_cond].keys():
            for cur_perf in specific_events[cur_cond][cur_side].keys():
                if len(specific_events[cur_cond][cur_side][cur_perf]):
                    for e in specific_events[cur_cond][cur_side][cur_perf]:
                        all_events.append(e)

    all_epochs = Epochs(eeg, all_events, tmin=epoch_tmin, tmax=epoch_length, baseline=baseline_corr, preload=True, event_repeated='merge')
    print('A total of {} epochs were extracted and baseline corrected.'.format(len(all_epochs)))

    if clean:
        ar = AutoReject()
        all_epochs, reject_log = ar.fit_transform(all_epochs, return_log=True) 
        reject_log.plot('horizontal')

    epochs_dict = dict()
    for cur_cond in specific_events.keys():
        epochs_dict[cur_cond] = dict()

        for cur_side in specific_events[cur_cond].keys():
            epochs_dict[cur_cond][cur_side] = dict()

            for cur_perf in specific_events[cur_cond][cur_side].keys():
                print('---------- {}-{}-{} -------------'.format(cur_cond, cur_side, cur_perf))

                if(len(specific_events[cur_cond][cur_side][cur_perf]) > 0):
                    #epochs_dict[cur_cond][cur_side][cur_perf] = Epochs(eeg, specific_events[cur_cond][cur_side][cur_perf], tmin=epoch_tmin, tmax=epoch_length, baseline=baseline_corr, preload=True, event_repeated='merge')
                    ids = [str(x) for x in set(specific_events[cur_cond][cur_side][cur_perf][:,2])]
                    epochs_dict[cur_cond][cur_side][cur_perf] = all_epochs[ids]
                else:
                    epochs_dict[cur_cond][cur_side][cur_perf] = None
    print("====================== /EPOCHING ======================")

    return epochs_dict


def get_CDA(epochs_dict, conds, sides, perfs, chan_right, chan_left):
    cda_dict = dict()

    for cur_cond in conds:
        cda_dict[cur_cond] = dict()

        for cur_side in sides:
            cda_dict[cur_cond][cur_side] = dict()

            for cur_perf in perfs:
                if epochs_dict[cur_cond][cur_side][cur_perf] is not None:
                    epochs_right = epochs_dict[cur_cond][cur_side][cur_perf].copy().pick_channels(chan_right)
                    epochs_left = epochs_dict[cur_cond][cur_side][cur_perf].copy().pick_channels(chan_left)

                    evoked_right = epochs_right.average()
                    evoked_left = epochs_left.average()

                    left = evoked_left.data.mean(0)
                    right = evoked_right.data.mean(0)

                    if cur_side == 'left':
                        cda_dict[cur_cond][cur_side][cur_perf] = left - right
                    else:
                        cda_dict[cur_cond][cur_side][cur_perf] = right - left
    
    return cda_dict


def get_CDA_perf_report(epochs_dict, conds, sides, perfs, chan_right, chan_left, cda_window):
    tmin = cda_window[0]
    tmax = cda_window[1]
    perf_report = dict()

    for cur_cond in conds:
        perf_report[cur_cond] = dict()

        for cur_side in sides:
            perf_report[cur_cond][cur_side] = dict()

            for cur_perf in perfs:
                if epochs_dict[cur_cond][cur_side][cur_perf] is not None:
                    epochs_right = epochs_dict[cur_cond][cur_side][cur_perf].copy().pick_channels(chan_right)
                    epochs_left = epochs_dict[cur_cond][cur_side][cur_perf].copy().pick_channels(chan_left)

                    perf_report[cur_cond][cur_side][cur_perf] = []
                    for i in range(0, len(epochs_right)):
                        left = epochs_left[i].get_data(tmin=tmin, tmax=tmax)[0].mean(0)
                        right = epochs_right[i].get_data(tmin=tmin, tmax=tmax)[0].mean(0)

                        if cur_side == 'left':
                            cur_perf_amp = (left - right).mean(0)
                            perf_report[cur_cond][cur_side][cur_perf].append(cur_perf_amp)
                            #print('Adding Mean Amp: {}'.format(cur_perf_amp))
                        else:
                            perf_report[cur_cond][cur_side][cur_perf].append((right - left).mean(0))
    
    return perf_report


def prep_report(conds, sides, perfs):
    columns = []
    columns.extend(conds)
    columns.extend(sides)
    columns.extend(perfs)

    for cond in conds:
        for side in sides:
            columns.append('{}-{}'.format(cond, side))
            columns.append('{}-{}-CDA'.format(cond, side))
            for perf in perfs:
                columns.append('{}-{}-{}'.format(cond, side, perf))
                columns.append('{}-{}-{}-CDA'.format(cond, side, perf))
                columns.append('{}-{}-CDA'.format(side, perf))
                if '{}-{}'.format(cond, perf) not in columns:
                    columns.append('{}-{}'.format(cond, perf))
                    columns.append('{}-{}-CDA'.format(cond, perf))
    
    columns.append('checksum')
    columns.append('notes')
    
    return pd.DataFrame(columns=columns)


def fill_report(report, filename, specific_events):
    cur_row = list()
    cur_row.extend([0] * (len(report.keys()) - 1)) # Excluding notes columns
    cur_row.extend([''])
    report = report.append(pd.DataFrame([cur_row], columns=report.columns, index=[filename]))
    
    # Looking for level 1: conditions
    for cond in specific_events.keys():
        if cond not in report.keys():
            raise ValueError('Event Key ({}) not in Report Columns.'.format(k))
        
        for side in specific_events[cond].keys():
            if side not in report.keys():
                raise ValueError('Event Key ({}) not in Report Columns.'.format(k))
            
            for perf in specific_events[cond][side].keys():
                if perf not in report.keys():
                    raise ValueError('Event Key ({}) not in Report Columns.'.format(k))
                
                #print("Adding: {}-{}-{} ({}) to: {}".format(cond, side, perf, len(specific_events[cond][side][perf]), cond))
                report.at[filename, cond] = report.at[filename, cond] + len(specific_events[cond][side][perf])
                report.at[filename, side] = report.at[filename, side] + len(specific_events[cond][side][perf])
                report.at[filename, perf] = report.at[filename, perf] + len(specific_events[cond][side][perf])
                report.at[filename, '{}-{}'.format(cond, side)] = report.at[filename, '{}-{}'.format(cond, side)] + len(specific_events[cond][side][perf])
                report.at[filename, '{}-{}'.format(cond, perf)] = report.at[filename, '{}-{}'.format(cond, perf)] + len(specific_events[cond][side][perf])
                report.at[filename, '{}-{}-{}'.format(cond, side, perf)] = report.at[filename, '{}-{}-{}'.format(cond, side, perf)] + len(specific_events[cond][side][perf])
    
    return report


def get_report(report, conds, sides, perfs, other_cols):
    sub_report_cols = []
    for cond in conds: sub_report_cols.append(cond)
    for side in sides: sub_report_cols.append(side)
    for perf in perfs: sub_report_cols.append(perf)
    
    for cond in conds: 
        for side in sides: 
            sub_report_cols.append('{}-{}'.format(cond,side))
        
    for cond in conds: 
        for side in sides: 
            for perf in perfs: 
                sub_report_cols.append('{}-{}-{}'.format(cond, side, perf))
    
    return report[sub_report_cols]


def add_cda_report(report, filename, conds, sides, perfs, cda_dict, times, cda_window):
    tmin = cda_window[0]
    tmax = cda_window[1]
    #sub_times = times[(times > tmin) & (times < tmax)]

    for cond in conds:
        for side in sides:
            for perf in perfs:
                if perf in cda_dict[cond][side].keys():
                    cur_cda = cda_dict[cond][side][perf][(times > tmin) & (times < tmax)]
                    cur_cda_mean =  cur_cda.mean() * 1e6
                    report.at[filename, '{}-{}-{}-CDA'.format(cond, side, perf)] = cur_cda_mean
                    print('[{}] Adding Mean CDA Amp for {}-{}-{}: {}'.format(filename, cond, side, perf, cur_cda_mean))
                else:
                    print('[{}] Skipping: {}-{}-{}'.format(filename, cond, side, perf))
                    report.at[filename, '{}-{}-{}-CDA'.format(cond, side, perf)] = float("NaN")

    # Mean CDA Amps for Perfs
    for perf in perfs:
        cda_means = []

        for side in sides:
            for cond in conds:
                cda_means.append(np.nanmean(report.at[filename, '{}-{}-{}-CDA'.format(cond,side,perf)]))

        report.at[filename,'{}-CDA'.format(perf)] = np.nanmean(cda_means)
        print('Perf: {}, Mean: {}'.format(perf, np.nanmean(cda_means)))
        print(cda_means)

    # Mean CDA Amps for Conds
    for cond in conds:
        for perf in perfs:
            cda_means = []
            for side in sides:
                cda_means.append(np.nanmean(report.at[filename, '{}-{}-{}-CDA'.format(cond,side,perf)]))

            report.at[filename,'{}-{}-CDA'.format(cond, perf)] = np.nanmean(cda_means)
            print('Cond: {} | Perf: {}, Mean: {}'.format(cond, perf, np.nanmean(cda_means)))
            print(cda_means)

    # Mean CDA Amps for Sides
    for side in sides:
        for perf in perfs:
            cda_means = []
            for cond in conds:
                cda_means.append(np.nanmean(report.at[filename, '{}-{}-{}-CDA'.format(cond,side,perf)]))

            report.at[filename,'{}-{}-CDA'.format(side, perf)] = np.nanmean(cda_means)
            print('Side: {} | Perf: {}, Mean: {}'.format(side, perf, np.nanmean(cda_means)))
            print(cda_means)

    return report


#==============================================
# Checksum -- Events (from Triggers) vs Epochs 
#==============================================
def checksum(specific_events, epochs_dict):
    checksum = 1

    print("Verifying Checksum (events vs epochs)...")
        
    for cond in specific_events.keys():
        for side in specific_events[cond].keys():
            for perf in specific_events[cond][side].keys():
                if (cond in epochs_dict.keys() and side in epochs_dict[cond].keys() and perf in epochs_dict[cond][side].keys()) and (cond in specific_events.keys() and side in specific_events[cond].keys() and perf in specific_events[cond][side].keys()):
                    if epochs_dict[cond][side][perf] is not None and specific_events[cond][side][perf] is not None:
                        if len(epochs_dict[cond][side][perf]) != len(specific_events[cond][side][perf]):
                            checksum = 0
                            print("Checksum FAILED! Length mismatch for: {}-{}-{} ({} events vs {} epochs)".format(cond, side, perf, len(specific_events[cond][side][perf]), len(epochs_dict[cond][side][perf])))
                    else:
                        if epochs_dict[cond][side][perf] is None and len(specific_events[cond][side][perf]) > 0:
                            checksum = 0
                            print("Checksum FAILED! Length mismatch for: {}-{}-{}".format(cond, side, perf))
                else:
                    checksum = 0
                    print("Checksum FAILED! Invalid key for: {}-{}-{}".format(cond, side, perf))

    if checksum == 1:    
        print("Checksum all good!")

    return checksum
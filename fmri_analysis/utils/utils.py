"""
some util functions
"""
import glob
import numpy as np
from os.path import join, dirname, basename, exists
import pandas as pd
from shutil import copyfile

def get_info(item,infile=None):
    """
    get info from settings file
    """
    filey = join('..','data_settings.txt')
    f = open(filey,'r') 
    infodict={}

    for l in f.read().splitlines():
    	key,val = l.split(':')
    	infodict[key]=val
    try:
        assert item in infodict
    except:
        raise Exception('infodict does not include requested item')
    return infodict[item]

def move_EV(subj, task):
    subj = subj.replace('sub-','')
    # get relevant directories
    behav_data = get_info('behav_data_directory')
    ev_data = join(behav_data,'event_files')
    fmri_data = get_info('fmri_data_directory')
    # get event file
    ev_file = glob.glob(join(ev_data,'*%s*%s*' % (subj, task)))[0]
    task_fmri_files = glob.glob(join(fmri_data,
                                     '*%s*' % subj,'*',
                                     'func','*%s*' % task))
    task_fmri_dir = dirname(task_fmri_files[0])
    base_name = basename(task_fmri_files[0]).split('_bold')[0]
    new_events_file = join(task_fmri_dir, base_name+'_events.tsv')
    copyfile(ev_file, new_events_file)
    return new_events_file
    
def move_EVs(overwrite=True):
    tasks = ['ANT', 'stroop']
    fmri_data = get_info('fmri_data_directory')
    created_files = []
    for subj_file in glob.glob(join(fmri_data,'sub*')):
        subj = basename(subj_file)
        for task in tasks:
            if overwrite==True or not exists(join(subj_file,'*',
                                                 'func', '*%s*' % task)):
                try:
                    name = move_EV(subj, task)
                    created_files.append(name)
                except IndexError:
                    print('Move_EV failed for the %s: %s' % (subj, task))
    return created_files

# How to model RT
# For each condition model responses with constant duration (average RT across subjects
# or block duration)
# RT as a separate regressor for each onset, constant duration, amplitude as parameteric
# regressor (function of RT)
def parse_EVs(events_df, task):
    def get_ev_vars(events_df, condition_list, col=None, 
                    amplitude = 1, duration = 0):
        # if a column is specified, group by the values in that column
        if col is not None:
            group_df = events_df.groupby(col)
            for condition, condition_name in condition_list:
                if type(condition) is not list:
                    condition = [condition]
                # get members of group identified by the condition list
                c_dfs = [group_df.get_group(c) for c in condition 
                         if c in group_df.groups.keys()]
                if len(c_dfs)!=0:
                    c_df = pd.concat(c_dfs)
                    conditions.append(condition_name)
                    onsets.append(c_df.onset.tolist())
                    if type(amplitude) in (int,float):
                        amplitudes.append([amplitude])
                    elif type(amplitude) == str:
                        amplitudes.append(c_df.loc[:,amplitude].tolist())
                    if type(duration) in (int,float):
                        durations.append([duration])
                    elif type(duration) == str:
                        durations.append(c_df.loc[:,duration].tolist())
        else:
            assert len(condition_list) == 1, \
                'If "col" is not specified, condition list must be an \
                 array of length 1 specifying the regressor name'
            group_df = events_df
            conditions.append(condition_list[0])
            onsets.append(group_df.onset.tolist())
            if type(amplitude) in (int,float):
                amplitudes.append([amplitude])
            elif type(amplitude) == str:
                amplitudes.append(group_df.loc[:,amplitude].tolist())
            if type(duration) in (int,float):
                durations.append([duration])
            elif type(duration) == str:
                durations.append(group_df.loc[:,duration].tolist())


    conditions = []
    onsets = []
    durations = []
    amplitudes = []
    if task == "stroop":
        get_ev_vars(events_df, [('congruent','congruent'), 
                                ('incongruent','incongruent')],
                    col='trial_type', duration='duration')
        get_ev_vars(events_df, ['response_time'], duration='duration', 
                    amplitude='response_time')
        get_ev_vars(events_df, [(0, 'error')], col='correct', 
                    duration='duration')
    return conditions, onsets, durations, amplitudes
    
def process_confounds(confounds_file):
    """
    scrubbing for TASK
    remove TRs where FD>.5, stdDVARS (that relates to DVARS>.5)
    regressors to use
    ['X','Y','Z','RotX','RotY','RotY','<-firsttemporalderivative','stdDVARs','FD','respiratory','physio','aCompCor0-5']
    junk regressor: errors, ommissions, maybe very fast RTs (less than 50 ms)
    """
    confounds_df = pd.read_csv(confounds_file, sep = '\t', 
                               na_values=['n/a']).fillna(0)
    excessive_movement = (confounds_df.FramewiseDisplacement>.5) & \
                            (confounds_df.stdDVARS>1.2)
    excessive_movement_TRs = excessive_movement[excessive_movement].index
    excessive_movement_regressors = np.zeros([confounds_df.shape[0], 
                                   np.sum(excessive_movement)])
    for i,TR in enumerate(excessive_movement_TRs):
        excessive_movement_regressors[TR,i] = 1
    excessive_movement_regressor_names = ['rejectTR_%d' % TR for TR in 
                                          excessive_movement_TRs]
    # get movement regressors
    movement_regressor_names = ['X','Y','Z','RotX','RotY','RotZ']
    movement_regressors = confounds_df.loc[:,movement_regressor_names]
    movement_regressor_names += ['Xtd','Ytd','Ztd','RotXtd','RotYtd','RotZtd']
    movement_deriv_regressors = np.gradient(movement_regressors,axis=0)
    # add additional relevant regressors
    add_regressor_names = ['FramewiseDisplacement', 'stdDVARS', 
                           'aCompCor0','aCompCor1','aCompCor2',
                           'aCompCor3','aCompCor4','aCompCor5'] 
    additional_regressors = confounds_df.loc[:,add_regressor_names].values
    regressors = np.hstack((movement_regressors,
                            movement_deriv_regressors,
                            additional_regressors,
                            excessive_movement_regressors))
    # concatenate regressor names
    regressor_names = movement_regressor_names + add_regressor_names + \
                      excessive_movement_regressor_names
    return regressors, regressor_names
        
        
    
    





















                                           
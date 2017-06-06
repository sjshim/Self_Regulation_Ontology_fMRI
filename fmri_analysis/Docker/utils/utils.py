"""
some util functions
"""
import glob
import numpy as np
from os.path import join, dirname, basename, exists
import pandas as pd
from shutil import copyfile

def get_contrasts(task):
    contrast_list = []
    if task == 'ANT':
        c1 = ['incongruent','T', ['incongruent'], [1]]
        c2 = ['congruent','T', ['congruent'], [1]]
        c3 = ['conflict_network','T', ['incongruent','congruent'], [1,-1]]
        c4 = ['spatial_cue','T', ['spatial_cue'], [1]]
        c5 = ['double_cue','T', ['double_cue'], [1]]
        c6 = ['orienting_network','T', ['spatial_cue','double_cue'], [1,-1]]
        c7 = ['response_time', 'T', ['response_time'], [1]]
        contrast_list = [c1,c2,c3,c4,c5,c6,c7]
    elif task == 'CCTHot':
        c1 = ['EV','T', ['EV'], [1]]
        c2 = ['risk','T', ['risk'], [1]]
        c3 = ['response_time', 'T', ['response_time'], [1]]
        contrast_list = [c1,c2,c3]
    elif task == 'DPX':
        c1 = ['AX','T', ['AX'], [1]]
        c2 = ['AY','T', ['AY'], [1]]
        c3 = ['BX','T', ['BX'], [1]]
        c4 = ['BY','T', ['BY'], [1]]
        c5 = ['BX-BY','T', ['BX','BY'], [1,-1]]
        c6 = ['AY-BY','T', ['AY','BY'], [1,-1]]
        c7 = ['AY-BX','T', ['AY','BX'], [1,-1]]
        c8 = ['BX-AY','T', ['AY','BX'], [1,-1]]
        c9 = ['response_time', 'T', ['response_time'], [1]]
        contrast_list = [c1,c2,c3,c4,c5,c6,c7,c8,c9]
    elif task == 'stroop':
        c1 = ['incongruent','T', ['incongruent'], [1]]
        c2 = ['congruent','T', ['congruent'], [1]]
        c3 = ['incongruent-congruent','T', ['incongruent','congruent'], [1,-1]]
        c4 = ['response_time', 'T', ['response_time'], [1]]
        contrast_list = [c1,c2,c3,c4]
    elif task == 'twoByTwo':
        c1 = ['cue_switch','T', ['cue_switch'], [1]]
        c2 = ['cue_stay','T', ['cue_stay'], [1]]
        c3 = ['task_switch','T', ['task_switch'], [1]]
        c4 = ['task_stay','T', ['task_stay'], [1]]
        c5 = ['cue_switch_cost','T', ['cue_switch','cue_stay'], [1,-1]]
        c6 = ['task_switch_cost','T', ['task_switch','task_stay'], [1,-1]]
        c7 = ['response_time', 'T', ['response_time'], [1]]
        contrast_list = [c1,c2,c3,c4,c5,c6,c7]
    elif task == 'WATT3':
        c1 = ['plan_PA_with','T', ['plan_PA_with'], [1]]
        c2 = ['plan_PA_without','T', ['plan_PA_without'], [1]]
        c3 = ['search_depth','T', ['plan_PA_with','plan_PA_without'], [1,-1]]
        contrast_list = [c1,c2,c3]
    return contrast_list
        
        
# How to model RT
# For each condition model responses with constant duration (average RT across subjects
# or block duration)
# RT as a separate regressor for each onset, constant duration, amplitude as parameteric
# regressor (function of RT)
def parse_EVs(events_df, task):
    def get_ev_vars(events_df, condition_list, col=None, 
                    amplitude = 1, duration = 0, subset=None,
                    onset_column='onset'):
        # if subset is specified as a string, use to query
        if subset is not None:
            events_df = events_df.query(subset)
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
                    onsets.append(c_df.loc[:,onset_column].tolist())
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
            onsets.append(group_df.loc[:,onset_column].tolist())
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
    if task == "ANT":
        get_ev_vars(events_df, [('spatial','spatial_cue'),
                                ('double', 'double_cue')],
                    col='cue', duration='duration')
        get_ev_vars(events_df, [('congruent','congruent'),
                                ('incongruent', 'incongruent')],
                    col='flanker_type', duration='duration')
        get_ev_vars(events_df, ['response_time'], duration='duration', 
                    amplitude='response_time')
        get_ev_vars(events_df, [(0, 'error')], col='correct', 
                    duration='duration')
    elif task == "CCTHot":
        get_ev_vars(events_df, ['EV'], duration='duration', 
                    amplitude='EV')
        get_ev_vars(events_df, ['risk'], duration='duration', 
                    amplitude='risk')
        get_ev_vars(events_df, ['num_click_in_round'], duration='duration', 
                    amplitude='num_click_in_round')
        get_ev_vars(events_df, [(1,'reward'), (0,'punishment')], col='feedback',
                    duration=0, amplitude=1)
        get_ev_vars(events_df, ['response_time'], duration='duration', 
                    amplitude='response_time')
    elif task == "DPX":
        get_ev_vars(events_df, [('AX','AX'), ('AY','AY'), 
                                ('BX', 'BX'), ('BY','BY')],
                    col='condition', duration='duration')
        get_ev_vars(events_df, ['response_time'], duration='duration', 
                    amplitude='response_time')
        get_ev_vars(events_df, [(0, 'error')], col='correct', 
                    duration='duration')
    elif task == "stroop":
        get_ev_vars(events_df, [('congruent','congruent'), 
                                ('incongruent','incongruent')],
                    col='trial_type', duration='duration')
        get_ev_vars(events_df, ['response_time'], duration='duration', 
                    amplitude='response_time')
        get_ev_vars(events_df, [(0, 'error')], col='correct', 
                    duration='duration')
    elif task == "twoByTwo":
        # cue switch contrasts
        get_ev_vars(events_df, [('switch','cue_switch'), 
                                ('stay','cue_stay')],
                    col='cue_switch', duration='duration')
        # task switch contrasts
        get_ev_vars(events_df, [('switch','task_switch'), 
                                ('stay','task_stay')],
                    col='task_switch', duration='duration')
        get_ev_vars(events_df, [(100,'CTI_100'), 
                                (900,'CTI_900')],
                    col='CTI', duration='duration')
        get_ev_vars(events_df, ['response_time'], duration='duration', 
                    amplitude='response_time')
        get_ev_vars(events_df, [(0, 'error')], col='correct', 
                    duration='duration')
    elif task == "WATT3":
        # planning conditions
        get_ev_vars(events_df, [('UA_with_intermediate','plan_UA_with'), 
                                ('UA_without_intermediate','plan_UA_without'),
                                ('PA_with_intermediate','plan_PA_with'),
                                ('PA_without_intermediate','plan_PA_without')],
                    col='condition', duration='duration', 
                    subset="planning==1")
        # move conditions
        get_ev_vars(events_df, [('UA_with_intermediate','move_UA_with'), 
                                ('UA_without_intermediate','move_UA_without'),
                                ('PA_with_intermediate','move_PA_with'),
                                ('PA_without_intermediate','move_PA_without')],
                    col='condition', duration='duration', 
                    subset="planning==0")
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
                           'aCompCor00','aCompCor01','aCompCor02',
                           'aCompCor03','aCompCor04','aCompCor05'] 
    additional_regressors = confounds_df.loc[:,add_regressor_names].values
    regressors = np.hstack((movement_regressors,
                            movement_deriv_regressors,
                            additional_regressors,
                            excessive_movement_regressors))
    # concatenate regressor names
    regressor_names = movement_regressor_names + add_regressor_names + \
                      excessive_movement_regressor_names
    return regressors, regressor_names
        
        
    
    





















                                           

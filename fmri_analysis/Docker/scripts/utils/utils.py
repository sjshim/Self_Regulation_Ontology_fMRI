"""
some util functions
"""
import glob
import numpy as np
from os.path import join, dirname, basename, exists
import pandas as pd
from shutil import copyfile

def get_contrasts(task, regress_rt=True):
    contrast_list = []
    if task == 'ANT':
        # contrasts vs baseline
        c1 = ['incongruent','T', ['incongruent'], [1]]
        c2 = ['congruent','T', ['congruent'], [1]]
        c3 = ['spatial_cue','T', ['spatial_cue'], [1]]
        c4 = ['double_cue','T', ['double_cue'], [1]]
        # contrasts 
        c5 = ['conflict_network','T', ['incongruent','congruent'], [1,-1]]
        c6 = ['orienting_network','T', ['spatial_cue','double_cue'], [1,-1]]
        contrast_list = [c1,c2,c3,c4,c5,c6]
        if regress_rt:
            c7 = ['response_time', 'T', ['response_time'], [1]]
            contrast_list.append(c7)
    elif task == 'CCTHot':
        # contrasts vs baseline
        c1 = ['EV','T', ['EV'], [1]]
        c2 = ['risk','T', ['risk'], [1]]
        contrast_list = [c1,c2]
        if regress_rt:
            c3 = ['response_time', 'T', ['response_time'], [1]]
            contrast_list.append(c3)
    elif task == 'discountFix':
        # contrasts vs baseline
        c1 = ['subjective_value','T', ['subjective_value'], [1]]
        c2 = ['larger_later','T', ['larger_later'], [1]]
        c3 = ['smaller_sooner','T', ['smaller_sooner'], [1]]
        # contrasts
        c4 = ['LL vs SS','T', ['larger_later','smaller_sooner'], [1,-1]]
        contrast_list = [c1,c2,c3,c4]
        if regress_rt:
            c5 = ['response_time', 'T', ['response_time'], [1]]
            contrast_list.append(c5)
    elif task == 'DPX':
        # contrasts vs baseline
        c1 = ['AX','T', ['AX'], [1]]
        c2 = ['AY','T', ['AY'], [1]]
        c3 = ['BX','T', ['BX'], [1]]
        c4 = ['BY','T', ['BY'], [1]]
        # contrasts 
        c5 = ['BX-BY','T', ['BX','BY'], [1,-1]]
        c6 = ['AY-BY','T', ['AY','BY'], [1,-1]]
        c7 = ['AY-BX','T', ['AY','BX'], [1,-1]]
        c8 = ['BX-AY','T', ['AY','BX'], [1,-1]]
        contrast_list = [c1,c2,c3,c4,c5,c6,c7,c8]
        if regress_rt:
            c9 = ['response_time', 'T', ['response_time'], [1]]
            contrast_list.append(c9)
    elif task == 'motorSelectiveStop':
        # contrasts vs baseline
        c1 = ['crit_go','T', ['crit_go'], [1]]
        c2 = ['crit_stop_success','T', ['crit_stop_success'], [1]]
        c3 = ['crit_stop_failure','T', ['crit_stop_failure'], [1]]
        c4 = ['noncrit_signal','T', ['noncrit_signal'], [1]]
        c5 = ['noncrit_nosignal','T', ['noncrit_nosignal'], [1]]
        # contrasts
        c6 = ['crit_stop_success-crit_go', 'T', 
                ['crit_stop_success', 'crit_go'], [1,-1]]
        c7 = ['crit_stop_failure-crit_go', 'T', 
                ['crit_stop_failure', 'crit_go'], [1,-1]]
        c8 = ['crit_go-noncrit_nosignal', 'T', 
                ['crit_go', 'noncrit_nosignal'], [1,-1]]
        c9 = ['noncrit_signal-noncrit_nosignal', 'T' ,
                ['noncrit_signal','noncrit_nosignal'], [1,-1]]
        c10 = ['crit_stop_success-crit_stop_failure','T',
                ['crit_stop_success', 'crit_stop_failure'], [1,-1]]
        c11 = ['crit_stop_failure-crit_stop_success','T', 
                ['crit_stop_failure', 'crit_stop_success'], [1,-1]]
        c12 = ['crit_stop_success-noncrit_signal','T',
                ['crit_stop_success', 'noncrit_signal'], [1,-1]]
        c13 = ['crit_stop_failure-noncrit_signal','T',
                ['crit_stop_failure', 'noncrit_signal'], [1,-1]]
        contrast_list = [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13]
    elif task == 'stopSignal':
        # contrasts vs baseline
        c1 = ['go','T', ['go'], [1]]
        c2 = ['stop_success','T', ['stop_success'], [1]]
        c3 = ['stop_failure','T', ['stop_failure'], [1]]
        # contrasts
        c4 = ['stop_success-go','T', ['stop_success', 'go'], [1,-1]]
        c5 = ['stop_failure-go','T', ['stop_failure', 'go'], [1,-1]]
        c6 = ['stop_success-stop_failure','T', 
                ['stop_success', 'stop_failure'], [1,-1]]
        c7 = ['stop_failure-stop_success','T', 
                ['stop_failure', 'stop_success'], [1,-1]]
        contrast_list = [c1,c2,c3,c4,c5,c6,c7]
    elif task == 'stroop':
        # contrasts vs baseline
        c1 = ['incongruent','T', ['incongruent'], [1]]
        c2 = ['congruent','T', ['congruent'], [1]]
        # contrasts
        c3 = ['incongruent-congruent','T', ['incongruent','congruent'], [1,-1]]
        contrast_list = [c1,c2,c3]
        if regress_rt:
            c4 = ['response_time', 'T', ['response_time'], [1]]
            contrast_list.append(c4)
    elif task == 'twoByTwo':
        # contrasts vs baseline
        c1 = ['cue_switch_100','T', ['cue_switch_100'], [1]]
        c2 = ['cue_stay_100','T', ['cue_stay_100'], [1]]
        c3 = ['task_switch_100','T', ['task_switch_100'], [1]]
        c4 = ['cue_switch_900','T', ['cue_switch_900'], [1]]
        c5 = ['cue_stay_900','T', ['cue_stay_900'], [1]]
        c6 = ['task_switch_900','T', ['task_switch_900'], [1]]
        # contrasts
        c5 = ['cue_switch_cost_100','T', 
                ['cue_switch_100','cue_stay_100'], [1,-1]]
        c6 = ['cue_switch_cost_900','T', 
                ['cue_switch_900','cue_stay_900'], [1,-1]]
        c7 = ['task_switch_cost_100','T', 
                ['task_switch_100','cue_switch_100'], [1,-1]]
        c8 = ['task_switch_cost_900','T', 
                ['task_switch_900','cue_switch_900'], [1,-1]]
        contrast_list = [c1,c2,c3,c4,c5,c6,c7,c8]
        if regress_rt:
            c9 = ['response_time', 'T', ['response_time'], [1]]
            contrast_list.append(c9)
    elif task == 'WATT3':
        # contrasts vs baseline
        c1 = ['plan_PA_with','T', ['plan_PA_with'], [1]]
        c2 = ['plan_PA_without','T', ['plan_PA_without'], [1]]
        # contrasts
        c3 = ['search_depth','T', ['plan_PA_with','plan_PA_without'], [1,-1]]
        contrast_list = [c1,c2,c3]
    return contrast_list
        
        
# How to model RT
# For each condition model responses with constant duration 
# (average RT across subjects or block duration)
# RT as a separate regressor for each onset, constant duration, 
# amplitude as parameteric regressor (function of RT)
def parse_EVs(events_df, task, regress_rt=True):
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
        if regress_rt == True:
            get_ev_vars(events_df, ['response_time'], duration='duration', 
                        amplitude='response_time')
        get_ev_vars(events_df, [(True, 'junk')], col='junk', 
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
        if regress_rt == True:
            get_ev_vars(events_df, ['response_time'], duration='duration', 
                        amplitude='response_time')
        get_ev_vars(events_df, [(True, 'junk')], col='junk', 
                    duration='duration')
    elif task == "discountFix":
        get_ev_vars(events_df, [('larger_later','larger_later'), 
                                ('smaller_sooner','smaller_sooner')],
                    col='choice', duration='duration')
        get_ev_vars(events_df, ['subjective_value'], duration='duration', 
                        amplitude='subjective_value')
        if regress_rt == True:
            get_ev_vars(events_df, ['response_time'], duration='duration', 
                        amplitude='response_time')
        get_ev_vars(events_df, [(True, 'junk')], col='junk', 
                    duration='duration')
    elif task == "DPX":
        get_ev_vars(events_df, [('AX','AX'), ('AY','AY'), 
                                ('BX', 'BX'), ('BY','BY')],
                    col='condition', duration='duration')
        if regress_rt == True:
            get_ev_vars(events_df, ['response_time'], duration='duration', 
                        amplitude='response_time')
        get_ev_vars(events_df, [(True, 'junk')], col='junk', 
                    duration='duration')
    elif task == "motorSelectiveStop":
        get_ev_vars(events_df, [('crit_go','crit_go'), 
                                ('crit_stop_success', 'crit_stop_success'), 
                                ('crit_stop_failure', 'crit_stop_failure'),
                                ('noncrit_signal', 'noncrit_signal'),
                                ('noncrit_nosignal', 'noncrit_nosignal')],
                    col='trial_type', duration='duration')
        get_ev_vars(events_df, [(True, 'junk')], col='junk', 
                    duration='duration')
    elif task == "stopSignal":
        get_ev_vars(events_df, [('go','go'), 
                                ('stop_success', 'stop_success'), 
                                ('stop_failure', 'stop_failure')],
                    col='trial_type', duration='duration')
        get_ev_vars(events_df, [(True, 'junk')], col='junk', 
                    duration='duration')
    elif task == "stroop":
        get_ev_vars(events_df, [('congruent','congruent'), 
                                ('incongruent','incongruent')],
                    col='trial_type', duration='duration')
        if regress_rt == True:
            get_ev_vars(events_df, ['response_time'], duration='duration', 
                        amplitude='response_time')
        get_ev_vars(events_df, [(True, 'junk')], col='junk', 
                    duration='duration')
    elif task == "twoByTwo":
        # cue switch contrasts
        get_ev_vars(events_df, [('switch','cue_switch_900'), 
                                ('stay','cue_stay_900')],
                    col='cue_switch', duration='duration',
                    subset="CTI==900")
        get_ev_vars(events_df, [('switch','cue_switch_100'), 
                                ('stay','cue_stay_100')],
                    col='cue_switch', duration='duration',
                    subset="CTI==100")
        # task switch contrasts
        get_ev_vars(events_df, [('switch','task_switch_900')],
                    col='task_switch', duration='duration',
                    subset="CTI==900")
        get_ev_vars(events_df, [('switch','task_switch_100')],
                    col='task_switch', duration='duration',
                    subset="CTI==100")
        if regress_rt == True:
            get_ev_vars(events_df, ['response_time'], duration='duration', 
                        amplitude='response_time')
        get_ev_vars(events_df, [(True, 'junk')], col='junk', 
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
        get_ev_vars(events_df, ['movement'], onset_column='movement_onset')
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
    add_regressor_names = ['FramewiseDisplacement', 'stdDVARS'] 
    add_regressor_names += [i for i in confounds_df.columns if 'aCompCor' in i]
    additional_regressors = confounds_df.loc[:,add_regressor_names].values
    regressors = np.hstack((movement_regressors,
                            movement_deriv_regressors,
                            additional_regressors,
                            excessive_movement_regressors))
    # concatenate regressor names
    regressor_names = movement_regressor_names + add_regressor_names + \
                      excessive_movement_regressor_names
    return regressors, regressor_names
        
def process_physio(cardiac_file, resp_file):
    cardiac_file = '/mnt/temp/sub-s130/ses-1/func/sub-s130_ses-1_task-stroop_run-1_recording-cardiac_physio.tsv.gz'
    resp_file = '/mnt/temp/sub-s130/ses-1/func/sub-s130_ses-1_task-stroop_run-1_recording-respiratory_physio.tsv.gz'
    
    





















                                           

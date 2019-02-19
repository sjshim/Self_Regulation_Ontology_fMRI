"""
some util functions
"""
import numpy as np
import pandas as pd

# ********************************************************
# 1st level analysis utility functions
# ********************************************************

def get_contrasts(task, regress_rt=True):
    contrast_list = []
    if task == 'ANT':
        # task
        c1 = ['task','T', ['task'], [1]]
        # contrasts vs baseline
        c2 = ['orienting_network','T', ['orienting_network'], [1]]
        c3 = ['conflict_network','T', ['conflict_network'], [1]]
        contrast_list = [c1,c2, c3]
        if regress_rt:
            c4 = ['response_time', 'T', ['response_time'], [1]]
            contrast_list.append(c4)
    elif task == 'CCTHot':
        # task
        c1 = ['task','T', ['task'], [1]]
        # contrasts vs baseline
        c2 = ['EV','T', ['EV'], [1]]
        c3 = ['risk','T', ['risk'], [1]]
        contrast_list = [c1,c2,c3]
        if regress_rt:
            c4 = ['response_time', 'T', ['response_time'], [1]]
            contrast_list.append(c4)
    elif task == 'discountFix':
        c1 = ['subjective_value','T', ['subjective_value'], [1]]
        c2 = ['LL_vs_SS','T', ['LL_vs_SS'], [1]]
        contrast_list = [c1,c2]
        if regress_rt:
            c3 = ['response_time', 'T', ['response_time'], [1]]
            contrast_list.append(c3)
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
        c8 = ['BX-AY','T', ['BX','AY'], [1,-1]]
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
        c1 = ['task', 'T', ['task'], [1]]
        c2 = ['incongruent-congruent','T', ['incongruent-congruent'], [1]]
        contrast_list = [c1, c2]
        if regress_rt:
            c3 = ['response_time', 'T', ['response_time'], [1]]
            contrast_list.append(c3)
    elif task == 'surveyMedley':
        # contrasts vs baseline
        c1 = ['stim_duration','T', ['stim_duration'], [1]]
        c2 = ['movement','T', ['movement'], [1]]
        contrast_list = [c1,c2]
        if regress_rt:
            c4 = ['response_time', 'T', ['response_time'], [1]]
            contrast_list.append(c4)
    elif task == 'twoByTwo':
        c1 = ['task', 'T', ['task'], [1]]
        # contrasts
        c2 = ['cue_switch_cost_100','T', 
                ['cue_switch_cost_100'], [1]]
        c3 = ['cue_switch_cost_900','T', 
                ['cue_switch_cost_900'], [1]]
        c4 = ['task_switch_cost_100','T', 
                ['task_switch_cost_100'], [1]]
        c5 = ['task_switch_cost_900','T', 
                ['task_switch_cost_900'], [1]]
        contrast_list = [c1,c2,c3,c4, c5]
        if regress_rt:
            c6 = ['response_time', 'T', ['response_time'], [1]]
            contrast_list.append(c6)
    elif task == 'WATT3':
        # contrasts vs baseline
        c1 = ['plan_PA_with','T', ['plan_PA_with'], [1]]
        c2 = ['plan_PA_without','T', ['plan_PA_without'], [1]]
        # contrasts
        c3 = ['search_depth','T', ['plan_PA_with','plan_PA_without'], [1,-1]]
        # nuisance
        c4 = ['movement', 'T', ['response_time'], [1]]
        c5 = ['feedback', 'T', ['response_time'], [1]]
        contrast_list = [c1,c2,c3,c4,c5]
        if regress_rt:
            c6 = ['response_time', 'T', ['response_time'], [1]]
            contrast_list.append(c6)
    # base covers generic contrast that just looks at all trials
    elif task == 'base':
        c1 = ['trial', 'T', ['trial'], [1]]
        contrast_list = [c1]
    return contrast_list
        
# functions to extract fmri events
      
def get_ev_vars(output_dict, events_df, condition_spec, col=None, 
                amplitude=1, duration=0, subset=None, onset_column='onset'):
    """ adds amplitudes, conditions, durations and onsets to an output_dict
    
    Args:
        events_df: events file to parse
        condition_spec: string specfying condition name, or list of tuples of the fomr
            (subset_key, name) where subset_key groups the rows in col. If a list,
            col must be specified
        col: the column to be subset by the keys in conditions
        amplitude: either an int or string. If int, sets a constant amplitude. If
            string, amplitude is set to that column
        duration: either an int or string. If int, sets a constant duration. If
            string, duration is set to that column
        subset: pandas query string to subset the data before use
        onset_column: the column of timing to be used for onsets
    
    """
    required_keys =  set(['amplitudes','conditions','durations','onsets'])
    assert set(output_dict.keys()) == required_keys
    amplitudes = output_dict['amplitudes']
    conditions = output_dict['conditions']
    durations = output_dict['durations']
    onsets = output_dict['onsets']
    
    # if subset is specified as a string, use to query
    if subset is not None:
        events_df = events_df.query(subset)
    # if amplitudes or durations were passed as a series, subset and conver tto list
    if type(duration) == pd.core.series.Series:
        duration = duration[events_df.index].tolist()
    if type(amplitude) == pd.core.series.Series:
        amplitude = amplitude[events_df.index].tolist()
        
    # if a column is specified, group by the values in that column
    if type(condition_spec) == list:
        assert (col is not None), "Must specify column when condition_spec is a list"
        group_df = events_df.groupby(col)
        for condition, condition_name in condition_spec:
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
                    amplitudes.append([amplitude]*len(onsets[-1]))
                elif type(amplitude) == str:
                    amplitudes.append(c_df.loc[:,amplitude].tolist())
                if type(duration) in (int,float):
                    durations.append([duration]*len(onsets[-1]))
                elif type(duration) == str:
                    durations.append(c_df.loc[:,duration].tolist())
    elif type(condition_spec) == str:
        group_df = events_df
        conditions.append(condition_spec)
        onsets.append(group_df.loc[:,onset_column].tolist())
        if type(amplitude) in (int,float):
            amplitudes.append([amplitude]*len(onsets[-1]))
        elif type(amplitude) == str:
            amplitudes.append(group_df.loc[:,amplitude].tolist())
        elif type(amplitude) == list:
            amplitudes.append(amplitude)
        if type(duration) in (int,float):
            durations.append([duration]*len(onsets[-1]))
        elif type(duration) == str:
            durations.append(group_df.loc[:,duration].tolist())
        elif type(duration) == list:
            durations.append(duration)
    # ensure that each column added is all numeric
    for attr in [durations, amplitudes, onsets]:
        assert np.issubdtype(np.array(attr[-1]).dtype, np.number)   
        assert pd.isnull(attr[-1]).sum() == 0
    

# specific task functions
def get_ANT_EVs(events_df, regress_rt=True):
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }
    # task
    get_ev_vars(output_dict, events_df,
                condition_spec='task',
                duration='duration',
                subset='junk==False')
    # cue type
    get_ev_vars(output_dict, events_df, 
                condition_spec=[('spatial', 'spatial'),
                                  ('double', 'double')],
                col='cue',
                duration='duration',
                subset='junk==False')
    # conflict type
    get_ev_vars(output_dict, events_df,
                condition_spec = [('congruent', 'congruent'),
                                  ('incongruent', 'incongruent')],
                col='flanker_type',
                duration='duration',
                subset='junk==False')
    # nuisance regressors
    get_ev_vars(output_dict, events_df, 
                condition_spec=[(True, 'junk')], 
                col='junk', 
                duration='duration')
    if regress_rt == True:
        get_ev_vars(output_dict, events_df, 
                    condition_spec='response_time', 
                    duration='duration', 
                    amplitude='response_time',
                    subset='junk==False')
    return output_dict


def get_CCTHot_EVs(events_df, regress_rt):
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }
    # task
    get_ev_vars(output_dict, events_df,
                condition_spec='task',
                duration='duration',
                subset='junk==False')
    # add main parametric regressors: EV and risk
    get_ev_vars(output_dict, events_df, 
                condition_spec='EV', 
                duration='duration', 
                amplitude='EV',
                subset='junk==False')
    get_ev_vars(output_dict, events_df, 
                condition_spec='risk',
                duration='duration', 
                amplitude='risk',
                subset='junk==False')
    # other regressors
    get_ev_vars(output_dict, events_df, 
                condition_spec='num_click_in_round', 
                duration='duration', 
                amplitude='num_click_in_round',
                subset='junk==False')
    get_ev_vars(output_dict, events_df, 
                condition_spec=[(1,'reward'), (0,'punishment')], 
                col='feedback',
                subset='junk==False')
    # nuisance regressors
    get_ev_vars(output_dict, events_df, 
                condition_spec=[(True, 'junk')], 
                col='junk', 
                duration='duration')
    if regress_rt == True:
        get_ev_vars(output_dict, events_df, 
                    condition_spec='response_time', 
                    duration='duration', 
                    amplitude='response_time',
                    subset='junk==False')
    return output_dict

def get_discountFix_EVs(events_df, regress_rt=True):
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }
    # regressors of interest
    trial_type = events_df.query('junk == False').trial_type \
                .replace({'larger_later': 1, 'smaller_sooner': -1})
    get_ev_vars(output_dict, events_df, 
                condition_spec='LL_vs_SS',
                amplitude=trial_type,
                duration='duration',
                subset='junk == False')
    get_ev_vars(output_dict, events_df, 
                condition_spec='subjective_value', 
                duration='duration', 
                amplitude='subjective_value',
                subset='junk==False')
    # nuisance regressors
    get_ev_vars(output_dict, events_df, 
                condition_spec=[(True, 'junk')],
                col='junk', 
                duration='duration')
    if regress_rt == True:
        get_ev_vars(output_dict, events_df, 
                    condition_spec='response_time', 
                    duration='duration', 
                    amplitude='response_time',
                    subset='junk==False')
    return output_dict

def get_DPX_EVs(events_df, regress_rt=True):
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }
    get_ev_vars(output_dict, events_df, 
                condition_spec=[('AX','AX'), ('AY','AY'), ('BX', 'BX'), ('BY','BY')],
                col='condition', 
                duration='duration',
                subset='junk==False')
    # nuisance regressors
    get_ev_vars(output_dict, events_df, 
                condition_spec=[(True, 'junk')], 
                col='junk', 
                duration='duration')  
    if regress_rt == True:
        get_ev_vars(output_dict, events_df, 
                    condition_spec='response_time', 
                    duration='duration', 
                    amplitude='response_time',
                    subset='junk==False')
    return output_dict

def get_motorSelectiveStop_EVs(events_df):
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }
    get_ev_vars(output_dict, events_df, 
                condition_spec=[('crit_go','crit_go'), 
                            ('crit_stop_success', 'crit_stop_success'), 
                            ('crit_stop_failure', 'crit_stop_failure'),
                            ('noncrit_signal', 'noncrit_signal'),
                            ('noncrit_nosignal', 'noncrit_nosignal')],
                col='trial_type', 
                duration='duration')
    # nuisance regressors
    get_ev_vars(output_dict, events_df, 
                condition_spec=[(True, 'junk')], 
                col='junk', 
                duration='duration')    
    return output_dict

def get_stopSignal_EVs(events_df):
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }
    get_ev_vars(output_dict, events_df, 
                condition_spec=[('go','go'), 
                            ('stop_success', 'stop_success'), 
                            ('stop_failure', 'stop_failure')],
                col='trial_type', 
                duration='duration',
                subset='junk==False')
    # nuisance regressors
    get_ev_vars(output_dict, events_df, 
                condition_spec=[(True, 'junk')], 
                col='junk', 
                duration='duration')    
    return output_dict

def get_stroop_EVs(events_df, regress_rt=True):
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }
    # task regressor
    get_ev_vars(output_dict, events_df,
                condition_spec='task',
                duration='duration',
                subset='junk==False')
    # contrast regressor
    #conflict_amplitudes = ((events_df.condition=='incongruent')*2-1)
    get_ev_vars(output_dict, events_df,
                condition_spec=[('incongruent', 'incongruent'),
                               ('congruent', 'congruent')],
                col='condition',
                duration='duration',
                subset='junk==False')
    
    # nuisance regressors
    get_ev_vars(output_dict, events_df, 
                condition_spec=[(True, 'junk')], 
                col='junk', 
                duration='duration') 
    if regress_rt == True:
        get_ev_vars(output_dict, events_df, 
                    condition_spec='response_time', 
                    duration='duration', 
                    amplitude='response_time',
                    subset='junk==False')
    return output_dict

def get_surveyMedley_EVs(events_df, regress_rt=True):
    output_dict = {
        'conditions': [],
        'onsets': [],
        'durations': [],
        'amplitudes': []
        }
    # nuisance regressors
    get_ev_vars(output_dict, events_df, 
                condition_spec='stim_duration', 
                duration='stim_duration')
    get_ev_vars(output_dict, events_df, 
                condition_spec='movement',  
                onset_column='movement_onset')
    get_ev_vars(output_dict, events_df, 
                condition_spec=[(True, 'junk')], 
                col='junk', 
                duration='duration')   
    if regress_rt == True:
        get_ev_vars(output_dict, events_df, 
                    condition_spec='response_time', 
                    duration='duration', 
                    amplitude='response_time',
                    subset='junk==False')
    return output_dict

    
def get_twoByTwo_EVs(events_df, regress_rt=True):
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }
    # task regressor
    get_ev_vars(output_dict, events_df,
                condition_spec='task',
                duration='duration',
                subset='junk==False')
    # cue switch contrasts
    events_df.cue_switch.fillna(1, inplace=True)
    cue_switches = events_df.cue_switch.replace({'switch':1,'stay':-1})
    get_ev_vars(output_dict, events_df, 
                condition_spec='cue_switch_cost_900',
                amplitude=cue_switches,
                duration='duration',
                subset="CTI==900 and task_switch=='stay' and junk==False")
    get_ev_vars(output_dict, events_df, 
                condition_spec='cue_switch_cost_100',
                amplitude=cue_switches,
                duration='duration',
                subset="CTI==100 and task_switch=='stay' and junk==False")
    # task switch contrasts
    task_switches = events_df.task_switch.replace({'switch':1,'stay':-1})
    get_ev_vars(output_dict, events_df, 
                condition_spec='task_switch_cost_900',
                amplitude=task_switches,
                duration='duration',
                subset="CTI==900 and cue_switch!='stay' and junk==False")
    get_ev_vars(output_dict, events_df, 
                condition_spec='task_switch_cost_100',
                amplitude=task_switches,
                duration='duration',
                subset="CTI==100 and cue_switch!='stay' and junk==False")
    # nuisance regressors
    get_ev_vars(output_dict, events_df, 
                condition_spec=[(True, 'junk')], 
                col='junk', 
                duration='duration')   
    if regress_rt == True:
        get_ev_vars(output_dict, events_df, 
                    condition_spec='response_time', 
                    duration='duration', 
                    amplitude='response_time',
                    subset='junk==False')
    return output_dict

def get_WATT3_EVs(events_df, regress_rt=True):
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }
    # planning conditions
    get_ev_vars(output_dict, events_df, 
                condition_spec=[('PA_with_intermediate','plan_PA_with'),
                                ('PA_without_intermediate','plan_PA_without')],
                col='condition', 
                duration='duration', 
                subset="planning==1")
    # nuisance regressors
    get_ev_vars(output_dict, events_df, 
                condition_spec='movement', 
                onset_column='movement_onset')
    get_ev_vars(output_dict, events_df, 
                condition_spec='feedback', 
                duration='duration',
                subset="trial_id=='feedback'")
    
    if regress_rt == True:
        get_ev_vars(output_dict, events_df, 
                    condition_spec='response_time', 
                    duration='duration', 
                    amplitude='response_time',
                    subset="trial_id != 'feedback'")
    return output_dict

def get_base_EVs(events_df):
    output_dict = {
        'conditions': [],
        'onsets': [],
        'durations': [],
        'amplitudes': []
        }
    get_ev_vars(output_dict, events_df, 
                condition_spec='trial',
                duration='duration')
    get_ev_vars(output_dict, events_df, 
                condition_spec=[(True, 'junk')], 
                col='junk', 
                duration='duration')   
    return output_dict

def get_beta_series(events_df, regress_rt=True):
    output_dict = {
        'conditions': [],
        'onsets': [],
        'durations': [],
        'amplitudes': []
        }
    for i, row in events_df.iterrows():
        if row.junk == False:
            output_dict['conditions'].append('trial_%s' % str(i+1).zfill(3))
            output_dict['onsets'].append([row.onset])
            output_dict['durations'].append([row.duration])
            output_dict['amplitudes'].append([1])
    # nuisance regressors
    get_ev_vars(output_dict, events_df, 
                condition_spec=[(True, 'junk')], 
                col='junk', 
                duration='duration')   
    if regress_rt == True:
        get_ev_vars(output_dict, events_df, 
                    condition_spec='response_time', 
                    duration='duration', 
                    amplitude='response_time',
                    subset='junk==False')
    return output_dict
    
# How to model RT
# For each condition model responses with constant duration 
# (average RT across subjects or block duration)
# RT as a separate regressor for each onset, constant duration, 
# amplitude as parameteric regressor (function of RT)
def parse_EVs(events_df, task, regress_rt=True):
    if task == "ANT":
        EV_dict = get_ANT_EVs(events_df, regress_rt)
    elif task == "CCTHot": 
        EV_dict = get_CCTHot_EVs(events_df, regress_rt)
    elif task == "discountFix": 
        EV_dict = get_discountFix_EVs(events_df, regress_rt)
    elif task == "DPX":
        EV_dict = get_DPX_EVs(events_df, regress_rt)
    elif task == "motorSelectiveStop": 
        EV_dict = get_motorSelectiveStop_EVs(events_df)
    elif task == 'surveyMedley':
        EV_dict = get_surveyMedley_EVs(events_df, regress_rt)
    elif task == "stopSignal":
        EV_dict = get_stopSignal_EVs(events_df)
    elif task == "stroop":
        EV_dict = get_stroop_EVs(events_df, regress_rt)
    elif task == "twoByTwo":
        EV_dict = get_twoByTwo_EVs(events_df, regress_rt)
    elif task == "WATT3":
        EV_dict = get_WATT3_EVs(events_df, regress_rt)
    # covers generic conversion of events_df into trial design file
    elif task == 'beta':
        EV_dict = get_beta_series(events_df, regress_rt)
    return EV_dict

    
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
    
    

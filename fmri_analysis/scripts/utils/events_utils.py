"""
some util functions
"""
import numpy as np
import pandas as pd


# functions to extract fmri events
def get_ev_vars(output_dict, events_df, condition_spec,
                col=None, amplitude=1, duration=0,
                subset=None, onset_column='onset', demean_amp=False):
    """ adds amplitudes, conditions, durations and onsets to an output_dict

    Args:
        events_df: events file to parse
        condition_spec: string specfying condition name, or
            list of tuples of the form (subset_key, name) where subset_key are
            one or more groups in col. If a list, col must be specified.
        col: the column to be subset by the keys in conditions
        amplitude: either an int or string. If int, sets a constant amplitude.
            If string, ...
        duration: either an int or string. If int, sets a constant duration. If
            string, duration is set to that column
        subset: pandas query string to subset the data before use

    """
    # make sure NaNs or Nones aren't passed
    for param in [duration, amplitude]:
        assert param is not None
        if np.issubdtype(type(param), np.number):
            assert not np.isnan(param)

    required_keys = set(['amplitudes', 'conditions', 'durations', 'onsets'])
    assert set(output_dict.keys()) == required_keys
    amplitudes = output_dict['amplitudes']
    conditions = output_dict['conditions']
    durations = output_dict['durations']
    onsets = output_dict['onsets']

    demean = lambda x: x  # if not demeaning, just return the list
    if demean_amp:
        demean = lambda x: x - np.mean(x)  # otherwise, actually demean

    # if subset is specified as a string, use to query
    if subset is not None:
        events_df = events_df.query(subset)
    # if given a series, subset and convert to list
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
            if len(c_dfs) != 0:
                c_df = pd.concat(c_dfs)
                conditions.append(condition_name)
                onsets.append(c_df.loc[:, onset_column].tolist())
                if np.issubdtype(type(amplitude), np.number):  # don't demean a constant
                    amplitudes.append([amplitude]*len(onsets[-1]))
                elif type(amplitude) == str:
                    amplitudes.append(demean(c_df.loc[:, amplitude]).tolist())
                if np.issubdtype(type(duration), np.number):
                    durations.append([duration]*len(onsets[-1]))
                elif type(duration) == str:
                    durations.append(c_df.loc[:, duration].tolist())
    elif type(condition_spec) == str:
        group_df = events_df
        conditions.append(condition_spec)
        onsets.append(group_df.loc[:, onset_column].tolist())
        if np.issubdtype(type(amplitude), np.number):  # don't to demean a constant
            amplitudes.append([amplitude]*len(onsets[-1]))
        elif type(amplitude) == str:
            amplitudes.append(demean(group_df.loc[:, amplitude]).tolist())
        elif type(amplitude) == list:
            amplitudes.append(demean(amplitude).tolist())
        if np.issubdtype(type(duration), np.number):
            durations.append([duration]*len(onsets[-1]))
        elif type(duration) == str:
            durations.append(group_df.loc[:, duration].tolist())
        elif type(duration) == list:
            durations.append(duration)

    # ensure that each column added is all numeric
    for attr in [durations, amplitudes, onsets]:
        assert np.issubdtype(np.array(attr[-1]).dtype, np.number)
        assert pd.isnull(attr[-1]).sum() == 0

# specific task functions
def get_cuedTS_EVs(events_df, regress_rt=True, cond_rt=False, return_metadict=False):
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }
    meta_dict = {}
    
    for cond in ['cstay', 'cswitch_tswitch', 'cswitch_tstay']:
        get_ev_vars(output_dict, events_df,
                    condition_spec=cond,
                    duration=1,
                    subset="junk == False and trial_type == '%s'" % cond)

        if cond_rt:
            get_ev_vars(output_dict, events_df,
                        condition_spec=cond+'_RT',
                        duration=1,
                        amplitude='response_time',
                        subset="junk == False and trial_type == '%s'" % cond,
                        demean_amp=True)

    # trial regressor for task > baseline
    get_ev_vars(output_dict, events_df,
                condition_spec='task',
                duration=1,
                amplitude=1,
                subset='junk==False')

    # nuisance regressors
    get_ev_vars(output_dict, events_df,
                condition_spec=[(True, 'junk')],
                col='junk',
                duration=1)

    if regress_rt:
        get_ev_vars(output_dict, events_df,
                    condition_spec='response_time',
                    duration=1,
                    amplitude='response_time',
                    subset='junk==False',
                    demean_amp=True)

    if return_metadict:
        return(output_dict, meta_dict)
    else:
        return output_dict


def get_directedForgetting_EVs(events_df, regress_rt=True, cond_rt=False, return_metadict=False):
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }
    meta_dict = {}

    for cond in ['pos','neg','con']:
        get_ev_vars(output_dict, events_df,
                    condition_spec=cond,
                    duration=1,
                    subset="junk == False and trial_type == '%s'" % cond)

        if cond_rt:
            get_ev_vars(output_dict, events_df,
                        condition_spec=cond+'_RT',
                        duration=1,
                        amplitude='response_time',
                        subset="junk == False and trial_type == '%s'" % cond,
                        demean_amp=True)

    # nuisance regressors
    get_ev_vars(output_dict, events_df,
                condition_spec=[(True, 'junk')],
                col='junk',
                duration=1)
    if regress_rt:
        get_ev_vars(output_dict, events_df,
                   condition_spec="response_time",
                   duration=1,
                   amplitude="response_time",
                   subset='junk == False',
                   demean_amp=True)

    if return_metadict:
        return(output_dict, meta_dict)
    else:
        return output_dict
    
def get_flanker_EVs(events_df, regress_rt=True, cond_rt=False, return_metadict=False):
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }
    meta_dict = {}
    
    # generate parametric regressors
    events_df['congruency_parametric'] = -1
    events_df.loc[events_df.trial_type == 'incongruent',
                  'congruency_parametric'] = 1

    get_ev_vars(output_dict, events_df,
                condition_spec='congruency_parametric',
                duration=1,
                amplitude='congruency_parametric',
                subset='junk==False',
                demean_amp=True)

    # Task>baseline regressor
    get_ev_vars(output_dict, events_df,
                condition_spec='task',
                col='trial_type',
                duration=1,
                subset='junk==False')

    # nuisance regressors
    get_ev_vars(output_dict, events_df,
                condition_spec=[(True, 'junk')],
                col='junk',
                duration=1)

    if regress_rt:
        get_ev_vars(output_dict, events_df,
                    condition_spec='response_time',
                    duration=1,
                    amplitude='response_time',
                    subset='junk==False',
                    demean_amp=True)
    if cond_rt:
        for cond in ['congruent', 'incongruent']:
            get_ev_vars(output_dict, events_df,
                        condition_spec=cond+'_RT',
                        duration=1,
                        amplitude='response_time',
                        subset="junk == False and trial_type == '%s'" % cond,
                        demean_amp=True)
    if return_metadict:
        return(output_dict, meta_dict)
    else:
        return output_dict
    
def get_goNogo_EVs(events_df, regress_rt=True, cond_rt=False, return_metadict=False):
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }
    meta_dict = {}

    #note: condition regressor
    conds = ['go', 'nogo_success']
    for cond in conds:
        get_ev_vars(output_dict, events_df,
                    condition_spec=cond,
                    duration=1,
                    subset="trial_type == '%s'" % cond)

        if cond_rt and ('success' not in cond):
            get_ev_vars(output_dict, events_df,
                        condition_spec=cond+'_RT',
                        duration=1,
                        amplitude='response_time',
                        subset="trial_type == '%s'" % cond,
                        demean_amp=True)
    # Task>baseline regressor
    get_ev_vars(output_dict, events_df,
                condition_spec='task',
                col='trial_type',
                duration=1)

    # nuisance regressors
    get_ev_vars(output_dict, events_df,
                condition_spec=[(True, 'junk')],
                col='junk',
                duration=1)

    if regress_rt:
        get_ev_vars(output_dict, events_df,
                    condition_spec='response_time',
                    duration=1,
                    amplitude='response_time',
                    demean_amp=True)
    if return_metadict:
        return(output_dict, meta_dict)
    else:
        return output_dict
    
def get_goNogo_nogo_failure_EVs(events_df, regress_rt=True, cond_rt=False, return_metadict=False):
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }
    meta_dict = {}

    #note: condition regressor

    conds = ['go', 'nogo_success', 'nogo_failure']

    for cond in conds:
        get_ev_vars(output_dict, events_df,
                    condition_spec=cond,
                    duration=1,
                    subset="trial_type == '%s'" % cond)
        if cond_rt and ('success' not in cond):
            get_ev_vars(output_dict, events_df,
                        condition_spec=cond+'_RT',
                        duration=1,
                        amplitude='response_time',
                        subset="trial_type == '%s'" % cond,
                        demean_amp=True)
    # Task>baseline regressor
    get_ev_vars(output_dict, events_df,
                condition_spec='task',
                col='trial_type',
                duration=1)

    # nuisance regressors
    get_ev_vars(output_dict, events_df,
                condition_spec=[(True, 'junk')],
                col='junk',
                duration=1)

    if regress_rt:
        get_ev_vars(output_dict, events_df,
                    condition_spec='response_time',
                    duration=1,
                    amplitude='response_time',
                    demean_amp=True)
    if return_metadict:
        return(output_dict, meta_dict)
    else:
        return output_dict

def get_nBack_EVs(events_df, regress_rt=True, cond_rt=False, return_metadict=False):
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }
    meta_dict = {}
    
    get_ev_vars(output_dict, events_df,
                condition_spec='one_back',
                duration=1,
                subset='junk==False and delay==1')
    
    get_ev_vars(output_dict, events_df,
                condition_spec='two_back',
                duration=1,
                subset='junk==False and delay==2')

    events_df['condition_parametric'] = -1
    events_df.loc[events_df.trial_type == 'match',
                  'condition_parametric'] = 1

    get_ev_vars(output_dict, events_df,
                condition_spec='condition_parametric',
                duration=1,
                amplitude='condition_parametric',
                subset='junk==False',
                demean_amp=True)


    # Task>baseline regressor
    get_ev_vars(output_dict, events_df,
                condition_spec='task',
                col='trial_type',
                duration=1,
                subset='junk==False')

    # nuisance regressors
    get_ev_vars(output_dict, events_df,
                condition_spec=[(True, 'junk')],
                col='junk',
                duration=1)

    if regress_rt:
        get_ev_vars(output_dict, events_df,
                    condition_spec='response_time',
                    duration=1,
                    amplitude='response_time',
                    subset='junk==False',
                    demean_amp=True)
    if cond_rt:
        for cond in ['match', 'mismatch']:
            get_ev_vars(output_dict, events_df,
                        condition_spec=cond+'_RT',
                        duration=1,
                        amplitude='response_time',
                        subset="junk == False and trial_type == '%s'" % cond,
                        demean_amp=True)
    if return_metadict:
        return(output_dict, meta_dict)
    else:
        return output_dict
    

def get_shapeMatching_EVs(events_df, regress_rt=True, cond_rt=False, return_metadict=False):
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }
    meta_dict = {}

    for cond in ['SSS', 'SDD', 'SNN', 'DSD', 'DDD', 'DDS', 'DNN']:
        get_ev_vars(output_dict, events_df,
                    condition_spec=cond,
                    duration=1,
                    subset="junk == False and trial_type == '%s'" % cond)

        if cond_rt:
            get_ev_vars(output_dict, events_df,
                        condition_spec=cond+'_RT',
                        duration=1,
                        amplitude='response_time',
                        subset="junk == False and trial_type == '%s'" % cond,
                        demean_amp=True)
    if regress_rt:
        get_ev_vars(output_dict, events_df,
                   condition_spec='response_time',
                   duration=1,
                   amplitude='response_time',
                   subset='junk == False',
                   demean_amp=True)
    if return_metadict:
        return(output_dict, meta_dict)
    else:
        return output_dict


def get_spatialTS_EVs(events_df, regress_rt=True, cond_rt=False, return_metadict=False):
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }
    meta_dict = {}

    for cond in ['tstay_cstay', 'tstay_cswitch', 'tswitch_cswitch']:
        get_ev_vars(output_dict, events_df,
                    condition_spec=cond,
                    duration=1,
                    subset="junk == False and trial_type == '%s'" % cond)

        if cond_rt:
            get_ev_vars(output_dict, events_df,
                        condition_spec=cond+'_RT',
                        duration=1,
                        amplitude='response_time',
                        subset="junk == False and trial_type == '%s'" % cond,
                        demean_amp=True)

    # trial regressor for task > baseline
    get_ev_vars(output_dict, events_df,
                condition_spec='task',
                duration=1,
                amplitude=1,
                subset='junk==False')

    # nuisance regressors
    get_ev_vars(output_dict, events_df,
                condition_spec=[(True, 'junk')],
                col='junk',
                duration=1)

    if regress_rt:
        get_ev_vars(output_dict, events_df,
                    condition_spec='response_time',
                    duration=1,
                    amplitude='response_time',
                    subset='junk==False',
                    demean_amp=True)

    if return_metadict:
        return(output_dict, meta_dict)
    else:
        return output_dict


def get_stopSignal_EVs(events_df, regress_rt=True, cond_rt=False, return_metadict=False):
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }
    meta_dict = {}

    for cond in ['go', 'stop_success', 'stop_failure']:
        get_ev_vars(output_dict, events_df,
                    condition_spec=cond,
                    duration=1,
                    subset="junk == False and trial_type == '%s'" % cond)

        if cond_rt and ('success' not in cond):
            get_ev_vars(output_dict, events_df,
                        condition_spec=cond+'_RT',
                        duration=1,
                        amplitude='response_time',
                        subset="junk == False and trial_type == '%s'" % cond,
                        demean_amp=True)
            
    # nuisance regressors
    get_ev_vars(output_dict, events_df,
                condition_spec=[(True, 'junk')],
                col='junk',
                duration=1)
    if regress_rt:
        get_ev_vars(output_dict, events_df,
                    condition_spec='response_time',
                    duration=1,
                    amplitude='response_time',
                    subset='junk==False',
                    demean_amp=True)
    if return_metadict:
        return(output_dict, meta_dict)
    else:
        return output_dict
    

def get_stopSignalWDirectedForgetting_EVs(events_df, regress_rt=True, cond_rt=False, return_metadict=False):
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }
    meta_dict = {}

    # parametric congruency regressor
    for cond in ['go_con', 'go_pos', 'go_neg', 
                 'stop_success_con', 'stop_success_pos', 'stop_success_neg', 
                 'stop_failure_con', 'stop_failure_pos', 'stop_failure_neg']:
        get_ev_vars(output_dict, events_df,
                    condition_spec=cond,
                    duration=1,
                    subset="junk == False and trial_type == '%s'" % cond)
        if cond_rt and ('success' not in cond):
            get_ev_vars(output_dict, events_df,
                        condition_spec=cond+'_RT',
                        duration=1,
                        amplitude='response_time',
                        subset="junk == False and trial_type == '%s'" % cond,
                        demean_amp=True)

    # trial regressor for task > baseline
    get_ev_vars(output_dict, events_df,
                condition_spec='task',
                duration=1,
                amplitude=1,
                subset='junk==False')

    # nuisance regressors
    get_ev_vars(output_dict, events_df,
                condition_spec=[(True, 'junk')],
                col='junk',
                duration=1)

    if regress_rt:
        get_ev_vars(output_dict, events_df,
                    condition_spec='response_time',
                    duration=1,
                    amplitude='response_time',
                    subset='junk==False',
                    demean_amp=True)

    if return_metadict:
        return(output_dict, meta_dict)
    else:
        return output_dict

def get_stopSignalWFlanker_EVs(events_df, regress_rt=True, cond_rt=False, return_metadict=False):
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }
    meta_dict = {}

    response_time = events_df.loc[events_df.junk == False,
                                  'response_time'].mean()
    meta_dict['task_RT'] = response_time

    for cond in ['go_congruent', 'go_incongruent', 
                 'stop_success_congruent', 'stop_success_incongruent', 
                 'stop_failure_congruent', 'stop_failure_incongruent']:
        get_ev_vars(output_dict, events_df,
                    condition_spec=cond,
                    duration=1,
                    subset="junk == False and trial_type == '%s'" % cond)
        if cond_rt and ('success' not in cond):
            get_ev_vars(output_dict, events_df,
                        condition_spec=cond+'_RT',
                        duration=1,
                        amplitude='response_time',
                        subset="junk == False and trial_type == '%s'" % cond,
                        demean_amp=True)
            
    # trial regressor for task > baseline
    get_ev_vars(output_dict, events_df,
                condition_spec='task',
                duration=1,
                amplitude=1,
                subset='junk==False')

    # nuisance regressors
    get_ev_vars(output_dict, events_df,
                condition_spec=[(True, 'junk')],
                col='junk',
                duration=1)

    if regress_rt:
        get_ev_vars(output_dict, events_df,
                    condition_spec='response_time',
                    duration=1,
                    amplitude='response_time',
                    subset='junk==False',
                    demean_amp=True)

    if return_metadict:
        return(output_dict, meta_dict)
    else:
        return output_dict

def get_directedForgettingWFlanker_EVs(events_df, regress_rt=True, cond_rt=False, return_metadict=False):
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }
    meta_dict = {}

    response_time = events_df.loc[events_df.junk == False,
                                  'response_time'].mean()
    meta_dict['task_RT'] = response_time

    for cond in ['congruent_pos', 'congruent_neg', 'congruent_con',
                'incongruent_pos', 'incongruent_neg', 'incongruent_con']:
        get_ev_vars(output_dict, events_df,
                    condition_spec=cond,
                    duration=1,
                    subset="junk == False and trial_type == '%s'" % cond)
        if cond_rt:
            get_ev_vars(output_dict, events_df,
                        condition_spec=cond+'_RT',
                        duration=1,
                        amplitude='response_time',
                        subset="junk == False and trial_type == '%s'" % cond,
                        demean_amp=True)

    # trial regressor for task > baseline
    get_ev_vars(output_dict, events_df,
                condition_spec='task',
                duration=1,
                amplitude=1,
                subset='junk==False')

    # nuisance regressors
    get_ev_vars(output_dict, events_df,
                condition_spec=[(True, 'junk')],
                col='junk',
                duration=1)

    if regress_rt:
        get_ev_vars(output_dict, events_df,
                    condition_spec='response_time',
                    duration=1,
                    amplitude='response_time',
                    subset='junk==False',
                    demean_amp=True)

    if return_metadict:
        return(output_dict, meta_dict)
    else:
        return output_dict


# How to model RT
# For each condition model responses with constant duration
# (average RT across subjects or block duration)
# RT as a separate regressor for each onset, constant duration,
# amplitude as parameteric regressor (function of RT)
def parse_EVs(events_df, task, regress_rt=True, cond_rt=False, return_metadict=False):
    func_map = {
        'cuedTS': get_cuedTS_EVs,
        'directedForgetting': get_directedForgetting_EVs,
        'flanker': get_flanker_EVs,
        'goNogo': get_goNogo_EVs,
        'goNogo_nogo_failure': get_goNogo_nogo_failure_EVs,
        'nBack': get_nBack_EVs,
        #'rest': get_rest_EVs,
        'shapeMatching': get_shapeMatching_EVs,
        'spatialTS': get_spatialTS_EVs,
        'stopSignal': get_stopSignal_EVs,
        'stopSignalWDirectedForgetting': get_stopSignalWDirectedForgetting_EVs,
        'stopSignalWFlanker': get_stopSignalWFlanker_EVs,
        'directedForgettingWFlanker': get_directedForgettingWFlanker_EVs
    }
    return func_map[task](events_df,
                          regress_rt=regress_rt,
                          cond_rt=cond_rt,
                          return_metadict=return_metadict)

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
def get_flanker_EVs(events_df, regress_rt=True, return_metadict=False):
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

    # generate parametric regressors
    events_df['congruency_parametric'] = -1
    events_df.loc[events_df.trial_type == 'incongruent',
                  'congruency_parametric'] = 1

    get_ev_vars(output_dict, events_df,
                condition_spec='congruency_parametric',
                duration=response_time,
                amplitude='congruency_parametric',
                subset='junk==False',
                demean_amp=True)

    # Task>baseline regressor
    get_ev_vars(output_dict, events_df,
                condition_spec='task',
                col='trial_type',
                duration=response_time,
                subset='junk==False')

    # nuisance regressors
    get_ev_vars(output_dict, events_df,
                condition_spec=[(True, 'junk')],
                col='junk',
                duration=response_time)

    if regress_rt:
        get_ev_vars(output_dict, events_df,
                    condition_spec='response_time',
                    duration=response_time,
                    amplitude='response_time',
                    subset='junk==False',
                    demean_amp=True)
    print(output_dict)
    if return_metadict:
        return(output_dict, meta_dict)
    else:
        return output_dict
    
def get_goNogo_EVs(events_df, regress_rt=True, return_metadict=False):
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

    # generate parametric regressors
    events_df['condition_parametric'] = -1
    events_df.loc[events_df.trial_type == 'nogo',
                  'condition_parametric'] = 1

    get_ev_vars(output_dict, events_df,
                condition_spec='condition_parametric',
                duration=response_time,
                amplitude='condition_parametric',
                subset='junk==False',
                demean_amp=True)

    # Task>baseline regressor
    get_ev_vars(output_dict, events_df,
                condition_spec='task',
                col='trial_type',
                duration=response_time,
                subset='junk==False')

    # nuisance regressors
    get_ev_vars(output_dict, events_df,
                condition_spec=[(True, 'junk')],
                col='junk',
                duration=response_time)

    if regress_rt:
        get_ev_vars(output_dict, events_df,
                    condition_spec='response_time',
                    duration=response_time,
                    amplitude='response_time',
                    subset='junk==False',
                    demean_amp=True)
    if return_metadict:
        return(output_dict, meta_dict)
    else:
        return output_dict

def get_nBack_EVs(events_df, regress_rt=True, return_metadict=False):
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

    # generate parametric regressors
    events_df['condition_parametric'] = -1
    events_df.loc[events_df.trial_type == 'match',
                  'condition_parametric'] = 1

    get_ev_vars(output_dict, events_df,
                condition_spec='condition_parametric',
                duration=response_time,
                amplitude='condition_parametric',
                subset='junk==False',
                demean_amp=True)

    # Task>baseline regressor
    get_ev_vars(output_dict, events_df,
                condition_spec='task',
                col='trial_type',
                duration=response_time,
                subset='junk==False')

    # nuisance regressors
    get_ev_vars(output_dict, events_df,
                condition_spec=[(True, 'junk')],
                col='junk',
                duration=response_time)

    if regress_rt:
        get_ev_vars(output_dict, events_df,
                    condition_spec='response_time',
                    duration=response_time,
                    amplitude='response_time',
                    subset='junk==False',
                    demean_amp=True)
    if return_metadict:
        return(output_dict, meta_dict)
    else:
        return output_dict

def get_discountFix_EVs(events_df, regress_rt=True, return_metadict=False):
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
    # trial regressor for task > baseline
    get_ev_vars(output_dict, events_df,
                condition_spec='task',
                duration=response_time,
                amplitude=1,
                subset='junk==False')

    # choice regressor
    events_df['choice_parametric'] = -1
    events_df.loc[events_df.trial_type == 'larger_later',
                  'choice_parametric'] = 1

    get_ev_vars(output_dict, events_df,
                condition_spec='choice',
                duration=response_time,
                amplitude='choice_parametric',
                subset='junk==False',
                demean_amp=True)

    # nuisance regressors
    get_ev_vars(output_dict, events_df,
                condition_spec=[(True, 'junk')],
                col='junk',
                duration=response_time)

    if regress_rt:
        get_ev_vars(output_dict, events_df,
                    condition_spec='response_time',
                    duration=response_time,
                    amplitude='response_time',
                    subset='junk==False',
                    demean_amp=True)

    if return_metadict:
        return(output_dict, meta_dict)
    else:
        return output_dict


def get_directedForgetting_EVs(events_df, regress_rt=True, return_metadict=False):
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }
    meta_dict = {}

    # task rt used for junk trials
    task_rt = events_df.loc[events_df.junk == False,
                                  'response_time'].mean()

    for cond in ['pos', 'neg', 'con']:
        rt = events_df.loc[(events_df.junk == False) &
                            (events_df.condition == cond),
                            'response_time'].mean()
        meta_dict['%s_RT' % cond] = rt
        get_ev_vars(output_dict, events_df,
                    condition_spec=cond,
                    duration=rt,
                    subset="junk == False and condition == '%s'" % cond)

        if regress_rt:
            get_ev_vars(output_dict, events_df,
                        condition_spec=cond+'_RT',
                        duration=rt,
                        amplitude='response_time',
                        subset="junk == False and condition == '%s'" % cond,
                        demean_amp=True)

    # nuisance regressors
    get_ev_vars(output_dict, events_df,
                condition_spec=[(True, 'junk')],
                col='junk',
                duration=task_rt)

    if return_metadict:
        return(output_dict, meta_dict)
    else:
        return output_dict


def get_manipulation_EVs(events_df, regress_rt=True, return_metadict=False):
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
    get_ev_vars(output_dict, events_df,
                condition_spec=[('cue', 'task')],
                col='trial_id',
                duration=10,
                subset='trial_type!="no_stim" and junk==False')

    # nuisance regressors
    get_ev_vars(output_dict, events_df,
                condition_spec=[(True, 'junk')],
                col='junk',
                duration=response_time)

    events_df.which_cue = events_df.which_cue.replace('LATER', 1)
    events_df.which_cue = events_df.which_cue.replace('NOW', -1)
    # cue regressor
    get_ev_vars(output_dict, events_df,
                condition_spec=[('cue', 'cue')],
                col='trial_id',
                duration=response_time,
                amplitude='which_cue',
                subset='trial_type!="no_stim" and junk==False',
                demean_amp=True)

    get_ev_vars(output_dict, events_df,
                condition_spec='rating',
                duration=response_time,
                amplitude='response',
                subset='trial_type!="no_stim" and junk==False and trial_id=="current_rating"',
                demean_amp=True)

    if regress_rt:
        get_ev_vars(output_dict, events_df,
                    condition_spec='response_time',
                    duration=response_time,
                    amplitude='response_time',
                    subset='trial_type!="no_stim" and junk==False and trial_id=="current_rating"',
                    demean_amp=True)

    events_df.stim_type = events_df.stim_type.replace('neutral', -1)
    events_df.stim_type = events_df.stim_type.replace('valence', 1)
    get_ev_vars(output_dict, events_df,
                condition_spec=[('probe', 'probe')],
                col='trial_id',
                duration=response_time,
                amplitude='stim_type',
                subset='trial_type!="no_stim" and junk==False',
                demean_amp=True)

    if return_metadict:
        return(output_dict, meta_dict)
    else:
        return output_dict


def get_motorSelectiveStop_EVs(events_df, regress_rt=True, return_metadict=False):
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }
    meta_dict = {}

    for cond in ['crit_go', 'crit_stop_success',
                 'crit_stop_failure', 'noncrit_signal',
                 'noncrit_nosignal']:
        if 'success' in cond:
            rt = 1
        else:
            rt = events_df.loc[(events_df.junk == False) &
                               (events_df.trial_type == cond),
                               'response_time'].mean()
            meta_dict['%s_RT' % cond] = rt
        get_ev_vars(output_dict, events_df,
                    condition_spec=cond,
                    duration=rt,
                    subset="junk == False and trial_type == '%s'" % cond)

        if regress_rt and ('success' not in cond):
            get_ev_vars(output_dict, events_df,
                        condition_spec=cond+'_RT',
                        duration=rt,
                        amplitude='response_time',
                        subset="junk == False and trial_type == '%s'" % cond,
                        demean_amp=True)

    # nuisance regressors
    go_rt = events_df.loc[(events_df.junk == False) &
                          (events_df.trial_type == 'crit_go'),
                          'response_time'].mean()
    get_ev_vars(output_dict, events_df,
                condition_spec=[(True, 'junk')],
                col='junk',
                duration=go_rt)

    if return_metadict:
        return(output_dict, meta_dict)
    else:
        return output_dict


def get_stopSignal_EVs(events_df, regress_rt=True, return_metadict=False):
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }
    meta_dict = {}

    for cond in ['go', 'stop_success', 'stop_failure']:
        if 'success' in cond:
            rt = 1
        else:
            rt = events_df.loc[(events_df.junk == False) &
                               (events_df.trial_type == cond),
                               'response_time'].mean()
            meta_dict['%s_RT' % cond] = rt
        get_ev_vars(output_dict, events_df,
                    condition_spec=cond,
                    duration=rt,
                    subset="junk == False and trial_type == '%s'" % cond)

        if regress_rt and ('success' not in cond):
            get_ev_vars(output_dict, events_df,
                        condition_spec=cond+'_RT',
                        duration=rt,
                        amplitude='response_time',
                        subset="junk == False and trial_type == '%s'" % cond,
                        demean_amp=True)
            
    # nuisance regressors
    go_rt = events_df.loc[(events_df.junk == False) &
                          (events_df.trial_type == 'go'),
                          'response_time'].mean()
    get_ev_vars(output_dict, events_df,
                condition_spec=[(True, 'junk')],
                col='junk',
                duration=go_rt)
    if return_metadict:
        return(output_dict, meta_dict)
    else:
        return output_dict
    

def get_stroop_EVs(events_df, regress_rt=True, return_metadict=False):
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
    # parametric congruency regressor
    events_df['congruency_parametric'] = -1
    events_df.loc[events_df.trial_type == 'incongruent',
                  'congruency_parametric'] = 1

    get_ev_vars(output_dict, events_df,
                condition_spec='congruency_parametric',
                duration=response_time,
                amplitude='congruency_parametric',
                subset='junk==False',
                demean_amp=True)

    # trial regressor for task > baseline
    get_ev_vars(output_dict, events_df,
                condition_spec='task',
                duration=response_time,
                amplitude=1,
                subset='junk==False')

    # nuisance regressors
    get_ev_vars(output_dict, events_df,
                condition_spec=[(True, 'junk')],
                col='junk',
                duration=response_time)

    if regress_rt:
        get_ev_vars(output_dict, events_df,
                    condition_spec='response_time',
                    duration=response_time,
                    amplitude='response_time',
                    subset='junk==False',
                    demean_amp=True)

    if return_metadict:
        return(output_dict, meta_dict)
    else:
        return output_dict


def get_surveyMedley_EVs(events_df, regress_rt=True, return_metadict=False):
    # this function needs to be carefully looked over before being revived
    output_dict = {
        'conditions': [],
        'onsets': [],
        'durations': [],
        'amplitudes': []
        }
    meta_dict = {}  # unused currently

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
    if regress_rt:
        get_ev_vars(output_dict, events_df,
                    condition_spec='response_time',
                    duration='group_RT',
                    amplitude='response_time',
                    subset='junk==False',
                    demean_amp=True)

    # trial regressor for task > baseline
    get_ev_vars(output_dict, events_df,
                condition_spec='task',
                duration='duration',
                amplitude=1,
                subset='junk==False')

    if return_metadict:
        return(output_dict, meta_dict)
    else:
        return output_dict


def get_twoByTwo_EVs(events_df, regress_rt=True, return_metadict=False):
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }
    meta_dict = {}

    means = events_df.loc[events_df.junk==False,
                          ['response_time', 'CTI']].mean()
    duration = means['response_time'] + (means['CTI']/1000)
    meta_dict['task_RT'] = means['response_time']
    meta_dict['mean_CTI'] = means['CTI']/1000

    events_df.trial_type = ['cue_'+c if c is not np.nan else 'task_'+t
                            for c, t in zip(events_df.cue_switch,
                                            events_df.task_switch)]
    events_df.trial_type.replace('cue_switch', 'task_stay_cue_switch',
                                 inplace=True)

    # trial type contrasts
    get_ev_vars(output_dict, events_df,
                condition_spec=[('task_switch', 'task_switch_900'),
                                ('task_stay_cue_switch', 'task_stay_cue_switch_900'),
                                ('cue_stay', 'cue_stay_900')],
                col='trial_type',
                duration=duration,
                subset="CTI==900 and junk==False")
    get_ev_vars(output_dict, events_df,
                condition_spec=[('task_switch', 'task_switch_100'),
                                ('task_stay_cue_switch', 'task_stay_cue_switch_100'),
                                ('cue_stay', 'cue_stay_100')],
                col='trial_type',
                duration=duration,
                subset="CTI==100 and junk==False")
    # nuisance regressors
    get_ev_vars(output_dict, events_df,
                condition_spec=[(True, 'junk')],
                col='junk',
                duration=duration)

    if regress_rt:
        get_ev_vars(output_dict, events_df,
                    condition_spec='response_time',
                    duration=duration,
                    amplitude='response_time',
                    subset='junk==False',
                    demean_amp=True)

    if return_metadict:
        return(output_dict, meta_dict)
    else:
        return output_dict


def get_WATT3_EVs(events_df, regress_rt=True, return_metadict=False):
    output_dict = {
            'conditions': [],
            'onsets': [],
            'durations': [],
            'amplitudes': []
            }
    meta_dict = {}

    events_df.junk = events_df.junk.replace({0.0: False, 1.0: True})
    planning_rt = events_df.loc[(events_df.planning == 1) & (events_df.junk == False), 'response_time'].mean()
    acting_rt = events_df.loc[(events_df.planning == 0) & (events_df.junk == False) & (events_df.trial_id.isin(['to_hand', 'to_board'])), 'response_time'].mean()
    meta_dict['planning_RT'] = planning_rt
    meta_dict['acting_RT'] = acting_rt

    events_df.block_duration = events_df.block_duration/1000
    events_df.condition = events_df.condition.replace('PA_without_intermediate',
                                                      -1)
    events_df.condition = events_df.condition.replace('PA_with_intermediate',
                                                      1)

    # Planning regressors
    get_ev_vars(output_dict, events_df,
                condition_spec='planning_event',
                duration=planning_rt,
                subset="planning==1 and junk==False")

    get_ev_vars(output_dict, events_df,
                condition_spec='planning_parametric',
                duration=planning_rt,
                amplitude='condition',
                subset="planning==1 and junk==False")

    # Acting regressors
    get_ev_vars(output_dict, events_df,
                condition_spec='acting_event',
                duration=acting_rt,
                subset="planning==0 and junk==False")

    get_ev_vars(output_dict, events_df,
                condition_spec='acting_parametric',
                duration=acting_rt,
                amplitude='condition',
                subset="planning==0 and trial_id!='feedback' and junk==False")

    # RT regressors
    if regress_rt:
        get_ev_vars(output_dict, events_df,
                    condition_spec='planning_RT',
                    duration=planning_rt,
                    amplitude='response_time',
                    subset="planning==1 and junk==False",
                    demean_amp=True)

        get_ev_vars(output_dict, events_df,
                    condition_spec='acting_RT',
                    duration=acting_rt,
                    amplitude='response_time',
                    subset="planning==0 and trial_id!='feedback' and junk==False",
                    demean_amp=True)

    # Nuisance regressors
    get_ev_vars(output_dict, events_df,
                condition_spec=[(True, 'junk')],
                col='junk',
                duration=1)
    # practice
    new_df = pd.DataFrame(np.zeros((1, len(events_df.columns))),
                          columns=events_df.columns)
    new_df.trial_id = 'practice'
    new_df.duration = events_df.onset[0]
    events_df = pd.concat([new_df, events_df]).reset_index(drop=True)
    get_ev_vars(output_dict, events_df,
                condition_spec='practice',
                col='trial_id',
                amplitude=1,
                duration='duration',
                subset="trial_id=='practice'")

    # feedback
    get_ev_vars(output_dict, events_df,
                condition_spec='feedback',
                duration='block_duration',
                subset="trial_id=='feedback'")

    if return_metadict:
        return(output_dict, meta_dict)
    else:
        return output_dict


# DEPRECATED
def get_beta_series(events_df, regress_rt=True, return_metadict=False):
    output_dict = {
        'conditions': [],
        'onsets': [],
        'durations': [],
        'amplitudes': []
        }
    for i, row in events_df.iterrows():
        if row.junk is False:
            output_dict['conditions'].append('trial_%s' % str(i+1).zfill(3))
            output_dict['onsets'].append([row.onset])
            output_dict['durations'].append([row.duration])
            output_dict['amplitudes'].append([1])
    # nuisance regressors
    get_ev_vars(output_dict, events_df,
                condition_spec=[(True, 'junk')],
                col='junk',
                duration='duration')
    if regress_rt:
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
def parse_EVs(events_df, task, regress_rt=True, return_metadict=False):
    func_map = {
        #'cuedTS': get_cuedTS_EVs,
        #'directedForgetting': get_directedForgetting_EVs,
        'flanker': get_flanker_EVs,
        'goNogo': get_goNogo_EVs,
        'nBack': get_nBack_EVs,
        #'rest': get_rest_EVs,
        #'shapeMatching': get_shapeMatching_EVs,
        #'spatialTS': get_spatialTS_EVs,
        'stopSignal': get_stopSignal_EVs,
        #'stopSignalWDirectedForgetting': get_stopSignalWDirectedForgetting_EVs,
        #'stopSignalWFlanker': get_stopSignalWFlanker_EVs,
        #'directedForgettingWFlanker': get_directedForgettingWFlanker_EVs
        # DEPRECATED 'beta' covers generic conversion of events_df into trial design file
        #'beta': get_beta_series,
    }
    return func_map[task](events_df,
                          regress_rt=regress_rt,
                          return_metadict=return_metadict)

import numpy as np
import pandas as pd
from expanalysis.experiments.jspsych_processing import calc_discount_fixed_DV

# *********************************
# helper functions
# *********************************
def get_drop_columns(df, columns=None, use_default=True):
    default_cols = ['block_duration', 'correct_response', 'exp_stage', 
                    'feedback_duration', 'possible_responses', 
                   'rt', 'stim_duration', 'text', 'time_elapsed',
                   'timing_post_trial', 'trial_num']
    drop_columns = []
    if columns is not None:
        drop_columns = columns
    if use_default == True:
        drop_columns = set(default_cols) | set(drop_columns)
    drop_columns = set(df.columns) & set(drop_columns)
    return drop_columns
    
def get_junk_trials(df):
    junk = pd.Series(False, df.index)
    if 'correct' in df.columns:
        junk = np.logical_or(junk,np.logical_not(df.correct))
    if 'rt' in df.columns:
        junk = np.logical_or(junk,df.rt < 50)
    return junk
    
def get_movement_times(df):
    """
    time elapsed is evaluated at the end of a trial, so we have to subtract
    timing post trial and the entire block duration to get the time when
    the trial started. Then add the reaction time to get the time of movement
    """
    trial_time = df.time_elapsed - df.block_duration - df.timing_post_trial + \
                 df.rt
    return trial_time

def get_trial_times(df):
    """
    time elapsed is evaluated at the end of a trial, so we have to subtract
    timing post trial and the entire block duration to get the time when
    the trial started
    """
    trial_time = df.time_elapsed - df.block_duration - df.timing_post_trial
    return trial_time
   
def create_events(df, exp_id, duration=None):
    events_df = None
    lookup = {'attention_network_task': create_ANT_event,
              'columbia_card_task_fmri': create_CCT_event,
              'discount_fixed': create_discountFix_event,
              'dot_pattern_expectancy': create_DPX_event,
              'motor_selective_stop_signal': create_motorSelectiveStop_event,
              'stop_signal': create_stopSignal_event,
              'stroop': create_stroop_event,
              'survey_medley': create_survey_event,
              'twobytwo': create_twobytwo_event,
              'ward_and_allport': create_WATT_event}
    fun = lookup.get(exp_id)
    if fun is not None:
        events_df = fun(df, duration=duration)
    return events_df

def row_match(df,row_list):
    bool_list = pd.Series(True,index=df.index)
    for i in range(len(row_list)):
        bool_list = bool_list & (df.iloc[:,i] == row_list[i])
    return bool_list[bool_list].index
        
# *********************************
# Functions to create event files
# *********************************

def create_ANT_event(df, duration=None):
    columns_to_drop = get_drop_columns(df)
    events_df = df[df['time_elapsed']>0]
    # add junk regressor
    events_df.loc[:,'junk'] = get_junk_trials(df)
    # reorganize and rename columns in line with BIDs specifications
    events_df.insert(0,'response_time',events_df.rt-events_df.rt[events_df.rt>0].mean())
    if duration is None:
        events_df.insert(0,'duration',events_df.stim_duration)
    else:
        events_df.insert(0,'duration',duration)
    # time elapsed is at the end of the trial, so have to remove the block 
    # duration
    events_df.insert(0,'onset',get_trial_times(df))
    # convert milliseconds to seconds
    events_df.loc[:,['response_time','onset','duration']]/=1000
    # drop unnecessary columns
    events_df = events_df.drop(columns_to_drop, axis=1)
    return events_df

def create_CCT_event(df, duration=None):
    columns_to_drop = get_drop_columns(df, columns = ['cards_left', 
                                                      'clicked_on_loss_card',
                                                      'round_points', 
                                                      'which_round'])

    events_df = df[df['time_elapsed']>0]
    # add junk regressor
    events_df.loc[:,'junk'] = get_junk_trials(df)
    # reorganize and rename columns in line with BIDs specifications
    events_df.insert(0,'response_time',events_df.rt-events_df.rt[events_df.rt>0].mean())
    if duration is None:
        events_df.insert(0,'duration',events_df.stim_duration)
    else:
        events_df.insert(0,'duration',duration)
    # time elapsed is at the end of the trial, so have to remove the block 
    # duration
    events_df.insert(0,'onset',get_trial_times(df))
    # convert milliseconds to seconds
    events_df.loc[:,['response_time','onset','duration']]/=1000
    # add feedback columns
    events_df.loc[:,'feedback'] = events_df.clicked_on_loss_card \
                                    .apply(lambda x: not x)
    # drop unnecessary columns
    events_df = events_df.drop(columns_to_drop, axis=1)
    return events_df

def create_discountFix_event(df, duration=None):
    columns_to_drop = get_drop_columns(df)
    events_df = df[df['time_elapsed']>0]
    # add junk regressor
    events_df.loc[:,'junk'] = get_junk_trials(df)
   
    # reorganize and rename columns in line with BIDs specifications
    events_df.loc[:,'trial_type'] = events_df.choice
    events_df.insert(0,'response_time',events_df.rt-events_df.rt[events_df.rt>0].mean())

    if duration is None:
        events_df.insert(0,'duration',events_df.stim_duration)
    else:
        events_df.insert(0,'duration',duration)
    # time elapsed is at the end of the trial, so have to remove the block 
    # duration
    events_df.insert(0,'onset',get_trial_times(df))
    # convert milliseconds to seconds
    events_df.loc[:,['response_time','onset','duration']]/=1000

    #additional parametric regressors: 
    #subjective value
    worker_id = df.worker_id.unique()[0]
    discount_rate = calc_discount_fixed_DV(df)[0].get(worker_id).get('hyp_discount_rate_glm').get('value')
    events_df.insert(0, 'subjective_value', events_df.large_amount/(1+discount_rate*events_df.later_delay))    
    #inverse_delay
    events_df.insert(0, 'inverse_delay', 1/events_df.later_delay)
    # drop unnecessary columns
    events_df = events_df.drop(columns_to_drop, axis=1)
    return events_df
    
def create_DPX_event(df, duration=None):
    columns_to_drop = get_drop_columns(df)
    events_df = df[df['time_elapsed']>0]
    # add junk regressor
    events_df.loc[:,'junk'] = get_junk_trials(df)
    # reorganize and rename columns in line with BIDs specifications
    events_df.loc[:,'trial_type'] = events_df.condition
    events_df.insert(0,'response_time',events_df.rt-events_df.rt[events_df.rt>0].mean())
    # Cue-to-Probe time
    CPI=1000
    if duration is None:
        events_df.insert(0,'duration',events_df.stim_duration+CPI)
    else:
        events_df.insert(0,'duration',duration+CPI)
    # time elapsed is at the end of the trial, so have to remove the block 
    # duration. We also want the trial
    onsets = get_trial_times(df)-CPI 
    events_df.insert(0,'onset',onsets)
    # add motor onsets
    events_df.insert(0,'movement_onset',get_movement_times(df))
    # convert milliseconds to seconds
    events_df.loc[:,['response_time','onset',
                     'duration','movement_onset']]/=1000
    # drop unnecessary columns
    events_df = events_df.drop(columns_to_drop, axis=1)
    return events_df

def create_motorSelectiveStop_event(df, duration=None):
    columns_to_drop = get_drop_columns(df, columns = ['condition',
                                                      'SS_duration',
                                                      'SS_stimulus',
                                                      'SS_trial_type'])
    events_df = df[df['time_elapsed']>0]
    # add junk regressor
    events_df.loc[:,'junk'] = get_junk_trials(df)
    stops = events_df.query("stopped == True and condition == 'stop'").index
    events_df.loc[stops,'junk'] = False
    # create condition column
    crit_key = events_df.query('condition=="stop"') \
                .correct_response.unique()[0]
    noncrit_key = events_df.query('condition=="ignore"') \
                    .correct_response.unique()[0]
    condition_df = events_df.loc[:,['correct_response',
                                    'SS_trial_type','stopped']]
    condition = pd.Series(index=events_df.index)
    condition[row_match(condition_df, [crit_key,'go',False])] = 'crit_go'
    condition[row_match(condition_df, 
                        [crit_key,'stop',True])] = 'crit_stop_success'
    condition[row_match(condition_df, 
                        [crit_key,'stop',False])] = 'crit_stop_failure'
    condition[row_match(condition_df, 
                        [noncrit_key,'stop',False])] = 'noncrit_signal'
    condition[row_match(condition_df, 
                        [noncrit_key,'go',False])] = 'noncrit_nosignal'

    events_df.loc[:,'trial_type'] = condition
    events_df.insert(0,'response_time',events_df.rt-events_df.rt[events_df.rt>0].mean())
    if duration is None:
        events_df.insert(0,'duration',events_df.stim_duration)
    else:
        events_df.insert(0,'duration',duration)
    # time elapsed is at the end of the trial, so have to remove the block 
    # duration
    events_df.insert(0,'onset',get_trial_times(df))
    # convert milliseconds to seconds
    events_df.loc[:,['response_time','onset','duration']]/=1000
    # drop unnecessary columns
    events_df = events_df.drop(columns_to_drop, axis=1)
    return events_df
    
def create_stopSignal_event(df, duration=None):
    columns_to_drop = get_drop_columns(df, columns = ['condition',
                                                      'SS_duration',
                                                      'SS_stimulus',
                                                      'SS_trial_type'])
    events_df = df[df['time_elapsed']>0]
    # add junk regressor
    events_df.loc[:,'junk'] = get_junk_trials(df)
    stops = events_df.query("stopped == True and SS_trial_type == 'stop'").index
    events_df.loc[stops,'junk'] = False
    # reorganize and rename columns in line with BIDs specifications
    # create condition label
    SS_success_trials = events_df.query('SS_trial_type == "stop" \
                                        and stopped == True').index
    SS_fail_trials = events_df.query('SS_trial_type == "stop" \
                                        and stopped == False').index
    events_df.loc[:,'condition'] = 'go'
    events_df.loc[SS_success_trials,'condition'] = 'stop_success'
    events_df.loc[SS_fail_trials,'condition'] = 'stop_failure'
    events_df.loc[:,'trial_type'] = events_df.condition
    events_df.insert(0,'response_time',events_df.rt-events_df.rt[events_df.rt>0].mean())
    if duration is None:
        events_df.insert(0,'duration',events_df.stim_duration)
    else:
        events_df.insert(0,'duration',duration)
    # time elapsed is at the end of the trial, so have to remove the block 
    # duration
    events_df.insert(0,'onset',get_trial_times(df))
    # convert milliseconds to seconds
    events_df.loc[:,['response_time','onset','duration']]/=1000
    # drop unnecessary columns
    events_df = events_df.drop(columns_to_drop, axis=1)
    return events_df
 
def create_stroop_event(df, duration=None):
    columns_to_drop = get_drop_columns(df)
    events_df = df[df['time_elapsed']>0]
    # add junk regressor
    events_df.loc[:,'junk'] = get_junk_trials(df)
    # reorganize and rename columns in line with BIDs specifications
    events_df.loc[:,'trial_type'] = events_df.condition
    events_df.insert(0,'response_time',events_df.rt-events_df.rt[events_df.rt>0].mean())
    if duration is None:
        events_df.insert(0,'duration',events_df.stim_duration)
    else:
        events_df.insert(0,'duration',duration)
    # time elapsed is at the end of the trial, so have to remove the block 
    # duration
    events_df.insert(0,'onset',get_trial_times(df))
    # convert milliseconds to seconds
    events_df.loc[:,['response_time','onset','duration']]/=1000
    # drop unnecessary columns
    events_df = events_df.drop(columns_to_drop, axis=1)
    return events_df

def create_survey_event(df, duration=None):
    columns_to_drop = get_drop_columns(df, 
                                       use_default=False,
                                       columns = ['block_duration',
                                                  'key_press',
                                                  'options',
                                                  'response',
                                                  'rt',
                                                  'stim_duration'
                                                  'time_elapsed',
                                                  'timing_post_trial',
                                                  'trial_id',
                                                  'trial_type'])
    events_df = df[df['time_elapsed']>0]
    # add junk regressor
    junk = get_junk_trials(df)
    # response with a key outside of item responses
    wrong_response = df.apply(lambda x: str(x['key_press']) not in x['item_responses'], axis=1)
    events_df.loc[:, 'junk'] = np.logical_or(junk, wrong_response)
    # add duration and response regressor
    if duration is None:
        events_df.insert(0,'duration',events_df.stim_duration)
    else:
        events_df.insert(0,'duration',duration)
    events_df.insert(0,'response_time',events_df.rt-events_df.rt[events_df.rt>0].mean())
    # time elapsed is at the end of the trial, so have to remove the block 
    # duration
    events_df.insert(0,'onset',get_trial_times(df))
    # add motor onsets
    events_df.insert(0,'movement_onset',get_movement_times(df))
    # convert milliseconds to seconds
    events_df.loc[:,['response_time','onset','duration',
                     'movement_onset']]/=1000
    # drop unnecessary columns
    events_df = events_df.drop(columns_to_drop, axis=1)
    return events_df

def create_twobytwo_event(df, duration=None):
    columns_to_drop = get_drop_columns(df)
    events_df = df[df['time_elapsed']>0]
    # add junk regressor
    events_df.loc[:,'junk'] = get_junk_trials(df)
    # reorganize and rename columns in line with BIDs specifications
    events_df.insert(0,'response_time',events_df.rt-events_df.rt[events_df.rt>0].mean())
    if duration is None:
        events_df.insert(0,'duration',events_df.stim_duration+df.CTI)
    else:
        events_df.insert(0,'duration',duration+df.CTI)
    # time elapsed is at the end of the trial, so have to remove the block 
    # duration
    events_df.insert(0,'onset',get_trial_times(df)-df.CTI)
    # add motor onsets
    events_df.insert(0,'movement_onset',get_movement_times(df))
    # convert milliseconds to seconds
    events_df.loc[:,['response_time','onset',
                     'duration','movement_onset']]/=1000
    # drop unnecessary columns
    events_df = events_df.drop(columns_to_drop, axis=1)
    return events_df

def create_WATT_event(df, duration):
    columns_to_drop = get_drop_columns(df, columns = ['correct',
                                                      'current_position',
                                                      'goal_state',
                                                      'min_moves',
                                                      'num_moves_made',
                                                      'problem_time',
                                                      'start_state'])
    
    events_df = df[df['exp_stage']=='test']
    # add junk regressor
    events_df.loc[:,'junk'] = False
    # get planning, movement, and feedback index
    planning_moves = events_df.query('trial_id == "to_hand"' +
                                  'and num_moves_made==1').index
    other_moves = events_df.query('not (trial_id == "to_hand"' +
                                  'and num_moves_made==1)' +
                                  ' and trial_id != "feedback"').index   
    feedback = events_df.query('trial_id == "feedback"').index
    # add planning indicator
    events_df.insert(1,'planning',0)
    events_df.loc[planning_moves,'planning'] = 1
    
    # add durations for planning
    events_df.loc[planning_moves,'duration'] = duration['planning_time']
    # add mean centered rt
    planning_mean = events_df.loc[planning_moves,'rt'].mean()
    events_df.insert(0,'response_time',events_df.rt)
    events_df.loc[planning_moves,'response_time']-=planning_mean
    
    # add durations for planning
    events_df.loc[other_moves,'duration'] = duration['move_time']
    # add mean centered rt
    movement_mean = events_df.loc[other_moves,'rt'].mean()
    events_df.loc[other_moves,'response_time']-=movement_mean
    
    # add durations for feedback
    events_df.loc[feedback, 'duration'] = events_df.loc[feedback, 'block_duration']
    
    
    # time elapsed is at the end of the trial, so have to remove the block 
    # duration
    events_df.insert(0,'onset',get_trial_times(df))
    # add motor onsets
    events_df.insert(0,'movement_onset',get_movement_times(df))
    # convert milliseconds to seconds
    events_df.loc[:,['onset','duration',
                     'response_time','movement_onset']]/=1000
    # drop unnecessary columns
    events_df = events_df.drop(columns_to_drop, axis=1)
    return events_df


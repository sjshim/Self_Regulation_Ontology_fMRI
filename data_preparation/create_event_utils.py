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
              'dot_pattern_expectancy': create_DPX_event,
              'stop_signal': create_stopSignal_event,
              'stroop': create_stroop_event,
              'twobytwo': create_twobytwo_event,
              'ward_and_allport': create_WATT_event}
    fun = lookup.get(exp_id)
    if fun is not None:
        events_df = fun(df, duration=duration)
    return events_df

def create_ANT_event(df, duration=None):
    columns_to_drop = ['correct_response', 'exp_stage', 'possible_responses', 
                       'rt', 'stim_duration', 'text', 'time_elapsed',
                       'timing_post_trial', 'trial_id', 'trial_type']
    events_df = df[df['time_elapsed']>0]
    # reorganize and rename columns in line with BIDs specifications
    events_df.insert(0,'response_time',events_df.rt)
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
    columns_to_drop = ['block_duration', 'cards_left', 'clicked_on_loss_card',
                       'exp_stage', 'possible_responses', 
                       'rt', 'stim_duration', 'round_points', 'text',
                       'time_elapsed', 'timing_post_trial', 'trial_id', 
                       'trial_type', 'which_round']
    events_df = df[df['time_elapsed']>0]
    # reorganize and rename columns in line with BIDs specifications
    events_df.insert(0,'response_time',events_df.rt)
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
    
def create_DPX_event(df, duration=None):
    columns_to_drop = ['correct_response', 'exp_stage', 'possible_responses', 
                       'rt', 'stim_duration', 'text', 'time_elapsed',
                       'timing_post_trial', 'trial_id', 'trial_type']
    events_df = df[df['time_elapsed']>0]
    # reorganize and rename columns in line with BIDs specifications
    events_df.loc[:,'trial_type'] = events_df.condition
    events_df.insert(0,'response_time',events_df.rt)
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

def create_stopSignal_event(df, duration=None):
    columns_to_drop = ['condition', 'correct_response', 'exp_stage', 
                       'feedback_duration', 'possible_responses', 'rt',
                       'SS_duration', 'SS_stimulus', 'SS_trial_type',
                       'stim_duration', 'text', 'time_elapsed', 
                       'timing_post_trial', 'text', 'trial_id']
    events_df = df[df['time_elapsed']>0]
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
    events_df.insert(0,'response_time',events_df.rt)
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
    columns_to_drop = ['condition', 'correct_response', 'exp_stage', 
                       'feedback_duration', 'possible_responses', 'rt',
                       'stim_duration', 'text', 'time_elapsed', 
                       'timing_post_trial', 'text', 'trial_id']
    events_df = df[df['time_elapsed']>0]
    # reorganize and rename columns in line with BIDs specifications
    events_df.loc[:,'trial_type'] = events_df.condition
    events_df.insert(0,'response_time',events_df.rt)
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

def create_twobytwo_event(df, duration=None):
    columns_to_drop = ['correct_response', 'exp_stage', 
                       'feedback_duration', 'possible_responses', 'rt',
                       'stim_duration', 'text', 'time_elapsed', 
                       'timing_post_trial', 'trial_id', 'trial_type']
    events_df = df[df['time_elapsed']>0]
    # reorganize and rename columns in line with BIDs specifications
    events_df.insert(0,'response_time',events_df.rt)
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
    columns_to_drop = [ 'correct', 'current_position', 'exp_stage',
                       'goal_state', 'min_moves', 'num_moves_made', 
                       'possible_responses', 'problem_time', 'start_state',
                       'text', 'trial_id', 'trial_type']
    events_df = df[df['time_elapsed']>0]
    # add planning indicator
    first_moves = events_df.query('trial_id == "to_hand" and num_moves_made==1').index
    events_df.insert(1,'planning',0)
    events_df.loc[first_moves,'planning'] = 1
    
    # add durations for planning
    events_df.loc[first_moves,'duration'] = duration['planning_time']
    # add response time
    events_df.insert(0,'response_time',events_df.rt)
    
    # time elapsed is at the end of the trial, so have to remove the block 
    # duration
    events_df.insert(0,'onset',get_trial_times(df))
    events_df = events_df.drop(['rt', 'stim_duration','time_elapsed',
                                'timing_post_trial'], axis=1)
    # add motor onsets
    events_df.insert(0,'movement_onset',get_movement_times(df))
    # convert milliseconds to seconds
    events_df.loc[:,['onset','duration',
                     'response_time','movement_onset']]/=1000
    # drop unnecessary columns
    events_df = events_df.drop(columns_to_drop, axis=1)
    return events_df


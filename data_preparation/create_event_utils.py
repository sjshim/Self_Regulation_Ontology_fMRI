def get_trial_times(df):
    """
    time elapsed is evaluated at the end of a trial, so we have to subtract
    it to get onset time
    """
    trial_time = df.time_elapsed - df.block_duration - df.timing_post_trial
    return trial_time
    
def create_events(df, exp_id, duration=None):
    events_df = None
    lookup = {'attention_network_task': create_ANT_event,
              'dot_pattern_expectancy': create_DPX_event,
              'stroop': create_stroop_event,
              'twobytwo': create_twobytwo_event}
    fun = lookup.get(exp_id)
    if fun is not None:
        events_df = fun(df, duration=duration)
    return events_df

def create_ANT_event(df, duration=None):
    columns_to_drop = ['correct_response', 'exp_stage', 'possible_responses', 
                       'text', 'trial_id', 'trial_type']
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
    events_df = events_df.drop(['rt', 'stim_duration','time_elapsed',
                                'timing_post_trial'], axis=1)
    # convert milliseconds to seconds
    events_df.loc[:,['response_time','onset','duration']]/=1000
    # drop unnecessary columns
    events_df = events_df.drop(columns_to_drop, axis=1)
    return events_df
    
def create_DPX_event(df, duration=None):
    columns_to_drop = ['correct_response', 'exp_stage', 'possible_responses', 
                       'text', 'trial_id']
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
    events_df = events_df.drop(['rt', 'stim_duration','time_elapsed',
                                'timing_post_trial', 'trial_type'], axis=1)
    # convert milliseconds to seconds
    events_df.loc[:,['response_time','onset','duration']]/=1000
    # drop unnecessary columns
    events_df = events_df.drop(columns_to_drop, axis=1)
    return events_df
    
def create_stroop_event(df, duration=None):
    columns_to_drop = ['correct_response', 'exp_stage', 
                       'feedback_duration', 'possible_responses', 
                       'text', 'trial_id']
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
    events_df = events_df.drop(['condition','rt',
                                'stim_duration','time_elapsed',
                                'timing_post_Trial'], axis=1)
    # convert milliseconds to seconds
    events_df.loc[:,['response_time','onset','duration']]/=1000
    # drop unnecessary columns
    events_df = events_df.drop(columns_to_drop, axis=1)
    return events_df

def create_twobytwo_event(df, duration=None):
    columns_to_drop = ['correct_response', 'exp_stage', 
                       'feedback_duration', 'possible_responses', 
                       'text', 'trial_id', 'trial_type']
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
    events_df = events_df.drop(['rt', 'stim_duration','time_elapsed',
                                'timing_post_trial'], axis=1)
    # convert milliseconds to seconds
    events_df.loc[:,['response_time','onset','duration']]/=1000
    # drop unnecessary columns
    events_df = events_df.drop(columns_to_drop, axis=1)
    return events_df


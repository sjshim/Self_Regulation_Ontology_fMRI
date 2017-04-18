import pandas as pd

def create_event_file(df, exp_id):
    events_file = None
    lookup = {'stroop': create_stroop_event}
    fun = lookup.get(exp_id)
    if fun is not None:
        events_file = fun(df)
    return events_file
    
def create_stroop_event(df):
    columns_to_drop = ['block_duration', 'correct_response', 'exp_stage', 
                       'feedback_duration', 'possible_responses', 'text',
                       'trial_id', 'trial_type']
    events_df = df[df['exp_stage']!="practice"]
    events_df = events_df.drop(columns_to_drop, axis=1)
    # reorganize and rename columns in line with BIDs specifications
    events_df.insert(0,'trial_type',events_df.condition)
    events_df.insert(0,'response_time',events_df.rt)
    events_df.insert(0,'duration',.1)
    events_df.insert(0,'onset',events_df.time_elapsed)
    events_df = events_df.drop(['condition','rt','time_elapsed'], axis=1)
    # convert milliseconds to seconds
    events_df.loc[:,['response_time','onset','duration']]/=1000
    return events_df

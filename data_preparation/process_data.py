from expanalysis.experiments.processing import clean_data
from glob import glob
import os
import pandas as pd

from create_event_utils import create_events
# some DVs are defined in utils if they deviate from normal expanalysis
from utils import get_timing_correction

# set up dictionary of dataframes for each task
tasks = ['attention_network_task','columbia_card_task_fmri',
         'discount_fixed', 'dot_pattern_expectancy', 
         'motor_selective_stop_signal', 'stop_signal', 'stroop', 'survey_medley',
         'twobytwo', 'ward_and_allport']
task_dfs = {}
for task in tasks:
    task_dfs[task] = pd.DataFrame()
        
# clean data
for subj_file in glob('../Data/raw/*/*'):
    filey = os.path.basename(subj_file)
    cleaned_file_name = '_cleaned.'.join(filey.split('.'))
    event_file_name = '_events.'.join(filey.split('.')).replace('csv','tsv')
    cleaned_file_path = os.path.join('../Data/processed', cleaned_file_name)
    events_file_path = os.path.join('../Data/event_files', event_file_name)
    # if this file has already been cleaned, continue
    if os.path.exists(cleaned_file_path):
        df = pd.read_csv(cleaned_file_path)
        exp_id = df.experiment_exp_id.unique()[0]
    else:
        # else proceed
        df = pd.read_csv(subj_file)
        # set time_elapsed in reference to the last trigger of internal calibration
        start_time = df.query('trial_id == "fmri_trigger_wait"').iloc[-1]['time_elapsed']
        df.time_elapsed-=start_time
        # correct start time for problematic scans
        df.time_elapsed-=get_timing_correction(filey)
        # add exp_id to every row
        exp_id = df.iloc[-2].exp_id
        if exp_id == 'columbia_card_task_hot':
            exp_id = 'columbia_card_task_fmri'
        df.loc[:,'experiment_exp_id'] = exp_id
        # make sure there is a subject column
        if 'subject' not in df.columns:
            print('Added subject column for file: %s' % filey)
            df.loc[:,'subject'] = filey.split('_')[0]
        # change column from subject to worker_id
        df.rename(columns={'subject':'worker_id'}, inplace=True)
        # post process data, drop rows, etc.....
        drop_columns = ['view_history', 'stimulus', 'trial_index', 
                        'internal_node_id', 'test_start_block','exp_id']
        df = clean_data(df, exp_id=exp_id, drop_columns=drop_columns)
        # drop unnecessary rows 
        drop_dict = {'trial_type': ['text'], 
                     'trial_id': ['fmri_response_test', 'fmri_scanner_wait', 
                                  'fmri_trigger_wait', 'fmri_buffer']}
        for row, vals in drop_dict.items():
            df = df.query('%s not in  %s' % (row, vals))
        df.to_csv(cleaned_file_path, index=False)
    task_dfs[exp_id] = pd.concat([task_dfs[exp_id], df], axis=0)
        
# save group behavior
for task,df in task_dfs.items():
    df.to_csv('../Data/processed/group_data/%s.csv' % task, index=False)
    
# get 90th percentile reaction time for events files:
task_50th_rts = {task: df.rt.quantile(.5) for task,df in task_dfs.items()}
# ward and allport requires more complicated durations
test_df = task_dfs['ward_and_allport'].query('exp_stage == "test"')
plan_times = test_df.query('trial_id == "to_hand" and num_moves_made==1').rt
move_times = test_df.query('trial_id in ["to_hand", "to_board"]') \
                    .groupby(['worker_id','problem_id']).rt.sum()
move_times.index = plan_times.index
move_times-=plan_times
task_50th_rts['ward_and_allport'] = {'planning_time': plan_times.quantile(.5),
                                     'move_time': move_times.quantile(.5)}

# calculate event files
for subj_file in glob('../Data/raw/*/*'):   
    filey = os.path.basename(subj_file)
    cleaned_file_name = '_cleaned.'.join(filey.split('.'))
    event_file_name = '_events.'.join(filey.split('.')).replace('csv','tsv')
    cleaned_file_path = os.path.join('../Data/processed', cleaned_file_name)
    events_file_path = os.path.join('../Data/event_files', event_file_name)
    
    # get cleaned file
    df = pd.read_csv(cleaned_file_path)
    exp_id = df.experiment_exp_id.unique()[0]
    task_rt = task_50th_rts[exp_id]
    if not os.path.exists(events_file_path):
        # create event file for task contrasts
        events_df = create_events(df, exp_id, duration=task_rt)
        if events_df is not None:
            events_df.to_csv(events_file_path, sep='\t', index=False)
        else:
            print("Events file wasn't created for %s" % subj_file)

"""
exp_DVs = {}
# calculate DVs
for task_data in glob('../Data/processed/group_data/*csv'):
    df = pd.read_csv(task_data)
    exp_id = df.experiment_exp_id.unique()[0]
    print(exp_id)
    # Experiments whose analysis aren't defined in expanalysis
    if exp_id in ['attention_network_task']:
        # fmri ANT analysis identical to expanalysis except two conditions are dropped
        dvs,description = calc_ANT_DV(df, use_group_fun = False)
        DVs, valence = organize_DVs(dvs)
    else:
        DVs, valence, description = calc_exp_DVs(df, use_group_fun = False)
    exp_DVs[exp_id] = DVs
"""
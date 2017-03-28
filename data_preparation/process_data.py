from expanalysis.experiments.processing import clean_data, calc_exp_DVs, organize_DVs
from glob import glob
import os
import pandas as pd
# some DVs are defined in utils if they deviate from normal expanalysis
from utils import calc_ANT_DV

# set up dictionary of dataframes for each task
tasks = ['attention_network_task','columbia_card_task_fmri',
         'discount_fixed', 'dot_pattern_expectancy', 
         'motor_selective_stop_signal', 'stop_signal', 'stroop', 'survey_medley',
         'twobytwo', 'ward_and_allport']
task_dfs = {}
for task in tasks:
    task_path = '../Data/processed/group_data/%s.csv' % task
    if os.path.exists(task_path):
        task_dfs[task] = pd.read_csv(task_path)
    else:
        task_dfs[task] = pd.DataFrame()
        
# clean data
for subj_file in glob('../Data/raw/*/*'):
    filey = os.path.basename(subj_file)
    cleaned_file_name = '_cleaned.'.join(filey.split('.'))
    cleaned_file_path = os.path.join('../Data/processed', cleaned_file_name)
    # if this file has already been cleaned, continue
    if os.path.exists(cleaned_file_path):
        continue
    else:
        # else proceed
        df = pd.read_csv(subj_file)
        # set time_elapsed in reference to the last trigger of internal calibration
        start_time = df.query('trial_id == "fmri_trigger_wait"').iloc[-1]['time_elapsed']
        df.time_elapsed-=start_time
        # add exp_id to every row
        exp_id = df.iloc[-2].exp_id
        if exp_id == 'columbia_card_task_hot':
            exp_id = 'columbia_card_task_fmri'
        df.loc[:,'experiment_exp_id'] = exp_id
        # change column from subject to worker_id
        df.rename(columns={'subject':'worker_id'}, inplace=True)
        # post process data, drop rows, etc.....
        df = clean_data(df, exp_id=exp_id)
        # drop unnecessary rows 
        drop_dict = {'trial_type': ['text'], 
                     'trial_id': ['fmri_response_test', 'fmri_scanner_wait', 
                                  'fmri_trigger_wait', 'fmri_buffer']}
        for row, vals in drop_dict.items():
            df = df.query('%s not in  %s' % (row, vals))
        df.to_csv(cleaned_file_path)
        task_dfs[exp_id] = pd.concat([task_dfs[exp_id], df], axis=0)
        
# save group behavior
for task,df in task_dfs.items():
    df.to_csv('../Data/processed/group_data/%s.csv' % task)

# calculate DVs
for task_data in glob('../Data/processed/group_data/*'):
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


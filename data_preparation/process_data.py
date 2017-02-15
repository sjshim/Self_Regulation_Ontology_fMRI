from expanalysis.experiments.processing import clean_data
from glob import glob
import os
import pandas as pd

for subj_file in glob('../Data/raw/*/*'):
    filey = os.path.basename(subj_file)
    cleaned_file_name = '_cleaned.'.join(filey.split('.'))
    cleaned_file_path = os.path.join('../Data/processed', cleaned_file_name)
    # if this file has already been cleaned, continue
    if os.path.exists(cleaned_file_path):
        continue
    # else proceed
    df = pd.read_csv(subj_file)
    # set time_elapsed in reference to the last trigger of internal calibration
    start_time = df.query('trial_id == "fmri_trigger_wait"').iloc[-1]['time_elapsed']
    df.time_elapsed-=start_time
    # add exp_id to every row
    exp_id = df.iloc[-2].exp_id
    df.loc[:,'experiment_exp_id'] = exp_id
    if exp_id == 'columbia_card_task_hot':
        exp_id = 'columbia_card_task_fmri'
    # post process data, drop rows, etc...
    df = clean_data(df, exp_id=exp_id)
    # drop unnecessary rows 
    drop_dict = {'trial_type': ['text'], 
                 'trial_id': ['fmri_response_test', 'fmri_scanner_wait', 
                              'fmri_trigger_wait', 'fmri_buffer']}
    for row, vals in drop_dict.items():
        df = df.query('%s not in  %s' % (row, vals))
    df.to_csv(cleaned_file_path)
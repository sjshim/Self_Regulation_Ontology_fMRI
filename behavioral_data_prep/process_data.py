import argparse
from collections import defaultdict
from expanalysis.experiments.processing import clean_data
from glob import glob
import os
import pandas as pd
from create_event_utils import create_events
# some DVs are defined in utils if they deviate from normal expanalysis
from utils import get_name_map, get_timing_correction, get_median_rts
#for working in jupyter lab 

parser = argparse.ArgumentParser()
parser.add_argument('--clear', action='store_true')
parser.add_argument('--quiet', action='store_false')
args = parser.parse_args()
clear = args.clear
verbose = args.quiet

# if clear delete files first
if clear:
    if verbose: print("Clearing Data")
    file_dir = os.path.dirname(__file__)
    for f in glob(os.path.join(file_dir, '../behavioral_data/processed/*csv')):
        os.remove(f)
    for f in glob(os.path.join(file_dir, '../behavioral_data/event_files/*tsv')):
        os.remove(f)
    for f in glob(os.path.join(file_dir, '../behavioral_data/processed/group_data/*csv')):
        os.remove(f)


# set up map between file names and names of tasks
name_map = get_name_map()

task_dfs = defaultdict(pd.DataFrame)

#make directories

# clean data
if verbose: print("Processing Tasks")
for subj_file in glob('../behavioral_data/raw/*/*'):
    filey = os.path.basename(subj_file)
    cleaned_file_name = '_cleaned.'.join(filey.split('.'))
    event_file_name = '_events.'.join(filey.split('.')).replace('csv','tsv')
    cleaned_file_path = os.path.join('../behavioral_data/processed', cleaned_file_name)
    events_file_path = os.path.join('../behavioral_data/event_files', event_file_name)
    # if this file has already been cleaned, continue
    if os.path.exists(cleaned_file_path):
        df = pd.read_csv(cleaned_file_path)
        exp_id = df.experiment_exp_id.unique()[0] #gets the value of experiment_exp_id, and assigns it to exp_id
    else:
        # else proceed
        df = pd.read_csv(subj_file, engine='python')
        
         # get exp_id
        if 'exp_id' in df.columns:
            exp_id = df.iloc[-2].exp_id 
        else:
            exp_id = '_'.join(os.path.basename(subj_file).split('_')[1:]).rstrip('.csv')
        if (exp_id == 'manipulationTask') | (exp_id == 'cue_control_food'): #fixes formatting for manip 
            exp_id = 'manipulation_task'
            
        #fixes difference in rest scanner input 
        if (exp_id == 'rest') | (exp_id == 'uh2_video') | (exp_id == 'manipulation_task'): 
            df = df.replace(to_replace='scanner_wait', value = 'fmri_trigger_wait', regex=True)
            
        # set time_elapsed in reference to the last trigger of internal calibration
        print(filey, exp_id)
        start_time = df.query('trial_id == "fmri_trigger_wait"').iloc[-1]['time_elapsed'] 
        df.time_elapsed-=start_time 
        
        # correct start time for problematic scans
        df.time_elapsed-=get_timing_correction(filey)
       
        # make sure the file name matches the actual experiment
        assert name_map[exp_id] in subj_file, \
          print('file %s does not match exp_id: %s' % (subj_file, exp_id))
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
                        'internal_node_id', 'test_start_block','exp_id',
                        'trigger_times']
        df = clean_data(df, exp_id=exp_id, drop_columns=drop_columns)
        # drop unnecessary rows
        drop_dict = {'trial_type': ['text'],
                     'trial_id': ['fmri_response_test', 'fmri_scanner_wait',
                                  'fmri_trigger_wait', 'fmri_buffer', 'scanner_wait', 'scanner_rest', 
                                  'end']}
        for row, vals in drop_dict.items():
            df = df.query('%s not in  %s' % (row, vals))
        df.to_csv(cleaned_file_path, index=False)
    task_dfs[exp_id] = pd.concat([task_dfs[exp_id], df], axis=0)
if verbose: print("Saving Group Data")

# save group behavior
for task,df in task_dfs.items():
    df.to_csv('../behavioral_data/processed/group_data/%s.csv' % task, index=False)
# get 50th percentile reaction time for events files:
task_50th_rts = get_median_rts(task_dfs)
print(task_50th_rts)

if verbose: print("Creating Event Files")
# calculate event files
for subj_file in glob('../behavioral_data/raw/*/*.csv'):
    filey = os.path.basename(subj_file)
    cleaned_file_name = '_cleaned.'.join(filey.split('.'))
    event_file_name = '_events.'.join(filey.split('.')).replace('csv','tsv')
    cleaned_file_path = os.path.join('../behavioral_data/processed', cleaned_file_name)
    events_file_path = os.path.join('../behavioral_data/event_files', event_file_name)

  # get & save cleaned file
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

if verbose: print("Finished Processing")

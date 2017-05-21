"""
some util functions
"""
import glob
from os.path import join, dirname, basename, exists
import pandas as pd
from shutil import copyfile

def get_info(item,infile=None):
    """
    get info from settings file
    """
    filey = join('..','data_settings.txt')
    f = open(filey,'r') 
    infodict={}

    for l in f.read().splitlines():
    	key,val = l.split(':')
    	infodict[key]=val
    try:
        assert item in infodict
    except:
        raise Exception('infodict does not include requested item')
    return infodict[item]

def move_EV(subj, task):
    subj = subj.replace('sub-','')
    # get relevant directories
    behav_data = get_info('behav_data_directory')
    ev_data = join(behav_data,'event_files')
    fmri_data = get_info('fmri_data_directory')
    # get event file
    ev_file = glob.glob(join(ev_data,'*%s*%s*' % (subj, task)))[0]
    task_fmri_files = glob.glob(join(fmri_data,
                                     '*%s*' % subj,'*',
                                     'func','*%s*' % task))
    task_fmri_dir = dirname(task_fmri_files[0])
    base_name = basename(task_fmri_files[0]).split('_bold')[0]
    new_events_file = join(task_fmri_dir, base_name+'_events.tsv')
    copyfile(ev_file, new_events_file)
    return new_events_file
    
def move_EVs(overwrite=True):
    tasks = ['ANT', 'stroop']
    fmri_data = get_info('fmri_data_directory')
    created_files = []
    for subj_file in glob.glob(join(fmri_data,'sub*')):
        subj = basename(subj_file)
        for task in tasks:
            if overwrite==True or not exists(join(subj_file,'*',
                                                 'func', '*%s*' % task)):
                try:
                    name = move_EV(subj, task)
                    created_files.append(name)
                except IndexError:
                    print('Move_EV failed for the %s: %s' % (subj, task))
    return created_files

def parse_EVs(events_df, task):
    def get_ev_vars(events_df, col, condition_list, amplitude = 1, duration = 0):
        group_df = events_df.groupby(col)
        for condition, condition_name in condition_list:
            if type(condition) is not list:
                condition = [condition]
            # get members of group identified by the condition list
            c_dfs = [group_df.get_group(c) for c in condition if c in group_df.groups.keys()]
            if len(c_dfs)!=0:
                c_df = pd.concat(c_dfs)
                conditions.append(condition_name)
                onsets.append(c_df.onset.tolist())
                if type(amplitude) in (int,float):
                    amplitudes.append([amplitude])
                elif type(amplitude) == str:
                    amplitudes.append(c_df.loc[:,amplitude].tolist())
                if type(duration) in (int,float):
                    durations.append([duration])
                elif type(duration) == str:
                    durations.append(c_df.loc[:,duration].tolist())
            
            
            

    conditions = []
    onsets = []
    durations = []
    amplitudes = []
    if task == "stroop":
        get_ev_vars(events_df, 'trial_type', [('congruent','congruent'), 
                                              ('incongruent','incongruent')], duration='response_time')
        get_ev_vars(events_df, 'key_press', [(89,'index_finger'), 
                                              (71,'middle_finger'),
                                              (82, 'ring_finger')], duration='response_time')
        get_ev_vars(events_df, 'correct', [(0, 'error')], duration='block_duration')
    return conditions, onsets, durations, amplitudes
                                                
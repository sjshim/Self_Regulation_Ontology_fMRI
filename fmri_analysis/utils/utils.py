"""
some util functions
"""
import glob
import numpy as np
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
                                     'func','*%s*bold*' % task))
    task_fmri_dir = dirname(task_fmri_files[0])
    base_name = basename(task_fmri_files[0]).split('_bold')[0]
    new_events_file = join(task_fmri_dir, base_name+'_events.tsv')
    copyfile(ev_file, new_events_file)
    return new_events_file
    
def move_EVs(overwrite=True):
    tasks = ['ANT','CCTHot','discountFix','DPX','motorSelectiveStop',
            'stopSignal','stroop','twoByTwo','WATT3']
    fmri_data = get_info('fmri_data_directory')
    created_files = []
    for subj_file in glob.glob(join(fmri_data,'sub-s???')):
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


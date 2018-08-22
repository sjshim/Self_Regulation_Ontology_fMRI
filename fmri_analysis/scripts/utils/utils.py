"""
some util functions
"""
from collections import defaultdict
from glob import glob
import numpy as np
from os.path import join, sep
import pandas as pd


# ********************************************************
# Basic Help Methods
# ********************************************************

def get_event_dfs(data_dir, subj='', task=''):
    event_paths = glob(join(data_dir, '*%s*' % subj,
                                '*', 'func', '*%s*event*' % task))
    event_dfs = defaultdict(dict)
    for event_path in event_paths:
        subj = event_path.split(sep)[-4].replace('sub-','')
        task = event_path.split(sep)[-1].split('_')[-3][5:]
        event_df = pd.read_csv(event_path, sep='\t')
        event_dfs[subj][task] = event_df
    return event_dfs
    
def load_atlas(atlas_path, atlas_label_path=None):
    out = {}
    out['maps'] = atlas_path
    if atlas_label_path:
        file_data = np.loadtxt(atlas_label_path, 
                               dtype={'names': ('index', 'label'),
                                      'formats': ('i4', 'S50')})
        out['labels'] = [i[1].decode('UTF-8') for i in file_data]
    return out


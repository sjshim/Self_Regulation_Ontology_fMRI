#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 12:29:03 2018

@author: ian
"""

from glob import glob
from os.path import join
import pandas as pd
import pickle
from utils.display_utils import get_design_df, plot_design

data_dir = '/mnt/OAK/fmri_analysis/output/1stLevel/'
task = 'ANT'

paths = glob(join(data_dir, '*%s*' % task))
path = paths[0]
df = get_design_df(path)

plot_design(df)
contrast_paths = glob(join(data_dir, '*%s*' % task, 'contrasts.pkl'))
contrast_path = contrast_paths[0]
contrasts = pickle.load(open(contrast_path, 'rb'))

events_dir = '/mnt/OAK/fmri_analysis/fmriprep/fmriprep'
events_dirs = glob(join(events_dir, '*','*ses*','func','*%s*' % task))
event_path = event_paths[0]
events_df = pd.DataFrame.from_csv(event_path, sep='\t')

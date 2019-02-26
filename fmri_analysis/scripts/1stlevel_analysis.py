
# coding: utf-8

# ### Imports

# In[1]:


import argparse
from inspect import currentframe, getframeinfo
from glob import glob
import numpy as np
import os
from os.path import join
import pandas as pd
from pathlib import Path
import pickle
import sys
import nibabel as nib
from nistats.first_level_model import FirstLevelModel
from utils.firstlevel_utils import get_first_level_objs, make_first_level_obj, save_first_level_obj


# ### Parse Arguments
# These are not needed for the jupyter notebook, but are used after conversion to a script for production
# 
# - conversion command:
#   - jupyter nbconvert --to script --execute 1stlevel_analysis.ipynb

# In[2]:


parser = argparse.ArgumentParser(description='First Level Entrypoint script')
parser.add_argument('-data_dir', default='/data')
parser.add_argument('-derivatives_dir', default=None)
parser.add_argument('-working_dir', default=None)
parser.add_argument('--subject_ids', default=None, nargs="+")
parser.add_argument('--tasks', nargs="+", help="Choose from ANT, CCTHot, discountFix,                                     DPX, motorSelectiveStop, stopSignal,                                     stroop, surveyMedley, twoByTwo, WATT3")
parser.add_argument('--rt', action='store_true')
parser.add_argument('--beta', action='store_true')
parser.add_argument('--n_procs', default=16, type=int)
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--quiet', '-q', action='store_true')

if '-derivatives_dir' in sys.argv or '-h' in sys.argv:
    args = parser.parse_args()
else:
    args = parser.parse_args([])
    args.data_dir = '/data'
    args.derivatives_dir = '/data/derivatives'
    args.tasks = ['stroop']
    args.subject_ids = ['s358']
    args.rt=True
    args.n_procs=1
    # get all subjects
    all_subjs = sorted([i[-4:] for i in glob(os.path.join(args.data_dir, '*')) if 'sub-' in i])
    args.subject_ids = all_subjs


# In[3]:


if not args.quiet:
    def verboseprint(*args, **kwargs):
        print(*args, **kwargs)
else:
    verboseprint = lambda *a, **k: None # do-nothing function


# ### Initial Setup

# In[4]:


# Set Paths
derivatives_dir = args.derivatives_dir
fmriprep_dir = join(derivatives_dir, 'fmriprep', 'fmriprep')
data_dir = args.data_dir
first_level_dir = join(derivatives_dir,'1stlevel')
if args.working_dir is None:
    working_dir = join(derivatives_dir, '1stlevel_workingdir')
else:
    working_dir = join(args.working_dir, '1stlevel_workingdir')

# set task
if args.tasks is not None:
    tasks = args.tasks
else:
    tasks = ['ANT', 'CCTHot', 'discountFix',
            'DPX', 'motorSelectiveStop',
            'stopSignal', 'stroop',
            'twoByTwo', 'WATT3']

# list of subject identifiers
if args.subject_ids is None:
    subjects = sorted([i[-4:] for i in glob(os.path.join(args.data_dir, '*')) if 'sub-' in i])
else:
    subjects = args.subject_ids
    
# other arguments
regress_rt = args.rt
beta_series = args.beta
n_procs = args.n_procs
# TR of functional images
TR = .68


# In[5]:


# print
verboseprint('*'*79)
verboseprint('Tasks: %s\n, Subjects: %s\n, derivatives_dir: %s\n, data_dir: %s' % 
     (tasks, subjects, derivatives_dir, data_dir))
verboseprint('*'*79)


# # Set up Nodes

# ### Run analysis

# In[6]:


to_run = []
for subject_id in subjects:
    for task in tasks:
        files = get_first_level_objs(subject_id, task, first_level_dir, regress_rt=True)
        if len(files) == 0 or args.overwrite:
            subjinfo = make_first_level_obj(subject_id, task, fmriprep_dir, 
                                            data_dir, TR, regress_rt=regress_rt)
            if subjinfo is not None:
                to_run.append(subjinfo)


# ### Run model fit

# In[9]:


for subjinfo in to_run:
    verboseprint(subjinfo.ID)
    verboseprint('** fitting model')
    fmri_glm = FirstLevelModel(TR, 
                           subject_label = subjinfo.ID,
                           mask=subjinfo.mask,
                           noise_model='ar1',
                           standardize=False, 
                           hrf_model='spm',
                           drift_model='cosine',
                           period_cut=80,
                           n_jobs=1
                          )
    out = fmri_glm.fit(subjinfo.func, design_matrices=subjinfo.design)
    subjinfo.fit_model = out
    
    # run contrasts and save
    verboseprint('** computing contrasts')
    for name, contrast in subjinfo.contrasts:
        z_map = subjinfo.fit_model.compute_contrast(contrast, output_type='z_score')
        subjinfo.maps[name+'_zscore'] = z_map
    verboseprint('** saving')
    save_first_level_obj(subjinfo, first_level_dir)


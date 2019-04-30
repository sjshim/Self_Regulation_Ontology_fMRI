
# coding: utf-8

# ### Imports

# In[ ]:


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
import warnings
from utils.firstlevel_plot_utils import plot_design
from utils.firstlevel_utils import get_first_level_objs, make_first_level_obj, save_first_level_obj


# ### Parse Arguments
# These are not needed for the jupyter notebook, but are used after conversion to a script for production
# 
# - conversion command:
#   - jupyter nbconvert --to script --execute 1stlevel_analysis.ipynb

# In[ ]:


parser = argparse.ArgumentParser(description='First Level Entrypoint script')
parser.add_argument('-data_dir', default='/data')
parser.add_argument('-derivatives_dir', default=None)
parser.add_argument('-fmriprep_dir', default=None)
parser.add_argument('-working_dir', default=None)
parser.add_argument('--subject_ids', nargs="+")
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
    args.tasks = ['ANT', 'CCTHot', 'discountFix', 'DPX', 
                  'motorSelectiveStop', 'stopSignal', 
                  'stroop', 'twoByTwo', 'WATT3']
    args.subject_ids = ['s358']
    args.rt=True
    args.n_procs=1
    args.derivatives_dir = '/data/derivatives/'
    args.data_dir = '/data'
    args.fmriprep_dir = '/data/derivatives/fmriprep/fmriprep'


# In[ ]:


if not args.quiet:
    def verboseprint(*args, **kwargs):
        print(*args, **kwargs)
else:
    verboseprint = lambda *a, **k: None # do-nothing function


# ### Initial Setup

# In[ ]:


# Set Paths
derivatives_dir = args.derivatives_dir
if args.fmriprep_dir is None:
    fmriprep_dir = join(derivatives_dir, 'fmriprep', 'fmriprep')
else:
    fmriprep_dir = args.fmriprep_dir
data_dir = args.data_dir
first_level_dir = join(derivatives_dir,'1stlevel')
if args.working_dir is None:
    working_dir = join(derivatives_dir, '1stlevel_workingdir')
else:
    working_dir = join(args.working_dir, '1stlevel_workingdir')

# set tasks
if args.tasks is not None:
    tasks = args.tasks
else:
    tasks = ['ANT', 'CCTHot', 'discountFix',
            'DPX', 'motorSelectiveStop',
            'stopSignal', 'stroop',
            'twoByTwo', 'WATT3']

# list of subject identifiers
if not args.subject_ids:
    subjects = sorted([i[-4:] for i in glob(os.path.join(args.data_dir, '*')) if 'sub-' in i])
else:
    subjects = args.subject_ids
    
# other arguments
regress_rt = args.rt
beta_series = args.beta
n_procs = args.n_procs
# TR of functional images
TR = .68


# In[ ]:


# print
verboseprint('*'*79)
verboseprint('Tasks: %s\n, Subjects: %s\n, derivatives_dir: %s\n, data_dir: %s' % 
     (tasks, subjects, derivatives_dir, data_dir))
verboseprint('*'*79)


# # Set up Nodes

# ### Run analysis

# In[ ]:


to_run = []
for subject_id in subjects:
    for task in tasks:
        verboseprint('Setting up %s, %s' % (subject_id, task))
        files = get_first_level_objs(subject_id, task, first_level_dir, regress_rt=regress_rt)
        if len(files) == 0 or args.overwrite:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",category=DeprecationWarning)
                warnings.filterwarnings("ignore",category=UserWarning)
                subjinfo = make_first_level_obj(subject_id, task, fmriprep_dir, 
                                                data_dir, TR, regress_rt=regress_rt)
            if subjinfo is not None:
                to_run.append(subjinfo)


# ### Run model fit

# In[ ]:


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
    """
    # run contrasts
    verboseprint('** computing contrasts')
    for name, contrast in subjinfo.contrasts:
        z_map = subjinfo.fit_model.compute_contrast(contrast, output_type='z_score')
        subjinfo.maps[name+'_zscore'] = z_map
    """
    verboseprint('** saving')
    save_first_level_obj(subjinfo, first_level_dir, True)
    subjinfo.export_design(first_level_dir)
    subjinfo.export_events(first_level_dir)


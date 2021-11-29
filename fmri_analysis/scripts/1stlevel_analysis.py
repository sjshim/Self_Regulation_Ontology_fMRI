#!/usr/bin/env python
# coding: utf-8

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
from nilearn.glm.first_level import FirstLevelModel
import warnings
from utils.plot_utils import plot_design
from utils.firstlevel_utils import get_first_level_objs, make_first_level_obj, save_first_level_obj


# In[2]:


parser = argparse.ArgumentParser(description='First Level Entrypoint script')
parser.add_argument('-data_dir', default='/data')
parser.add_argument('-derivatives_dir', default=None)
parser.add_argument('-fmriprep_dir', default=None)
parser.add_argument('-working_dir', default=None)
parser.add_argument('--subject_ids', nargs="+")
parser.add_argument('--tasks', nargs="+", help="Choose from ANT, CCTHot, discountFix, DPX, motorSelectiveStop, stopSignal, stroop, surveyMedley, twoByTwo, WATT3")
parser.add_argument('--rt', action='store_true')
parser.add_argument('--beta', action='store_true')
parser.add_argument('--n_procs', default=16, type=int)
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--quiet', '-q', action='store_true')
parser.add_argument('--design_matrix_only', '-dm', action='store_true')
parser.add_argument('--a_comp_cor', action='store_true')
parser.add_argument('--use_aroma', action='store_true')

if '-derivatives_dir' in sys.argv or '-h' in sys.argv:
    args = parser.parse_args()
else:
    args = parser.parse_args([])
    args.tasks = ['discountFix', 'manipulationTask',  # aim 2 tasks
                  'motorSelectiveStop', 'stopSignal']
    # args.tasks = ['ANT', 'CCTHot', 'discountFix', 'DPX', #aim 1 tasks
    #               'motorSelectiveStop', 'stopSignal',
    #               'stroop', 'twoByTwo', 'WATT3']

    args.rt = True
    args.a_comp_cor = True
    args.use_aroma = True
    args.n_procs = 1
    args.derivatives_dir = '/data/derivatives/'
    args.data_dir = '/data'
    args.fmriprep_dir = '/data/derivatives/fmriprep/'
    args.design_matrix_only = False


# In[3]:


if not args.quiet:
    def verboseprint(*args, **kwargs):
        print(*args, **kwargs)
else:
    def verboseprint(*args, **kwards):  # do-nothing function
        pass


# In[4]:

# Set Paths
derivatives_dir = args.derivatives_dir
if args.fmriprep_dir is None:
    fmriprep_dir = join(derivatives_dir, 'fmriprep')
else:
    fmriprep_dir = args.fmriprep_dir
data_dir = args.data_dir
first_level_dir = join(derivatives_dir, '1stlevel')
if args.working_dir is None:
    working_dir = join(derivatives_dir, '1stlevel_workingdir')
else:
    working_dir = join(args.working_dir, '1stlevel_workingdir')

# set tasks
if args.tasks is not None:
    tasks = args.tasks
else:
    tasks = ['discountFix', 'manipulationTask',
             'motorSelectiveStop', 'stopSignal']

    # tasks = ['ANT', 'CCTHot', 'discountFix', 'DPX',
    #               'motorSelectiveStop', 'stopSignal',
    #               'stroop', 'twoByTwo', 'WATT3']


# list of subject identifiers
if not args.subject_ids:
    subjects = sorted([i.split("-")[-1] for i in glob(
                       os.path.join(args.data_dir, '*')) if 'sub-' in i])
else:
    subjects = args.subject_ids
    
#removing pilot subject
subjects.remove('n01')
#removing s03 for brainmask issue
subjects.remove('s03')   

ses = []
for i in range(1, 10):
    ses.append('ses-0'+str(i))
for i in range(10, 13):
    ses.append('ses-'+str(i))
# other arguments
regress_rt = args.rt
beta_series = args.beta
cond_rt = args.cond_rt
n_procs = args.n_procs

# TR of functional images
TR = 1.49
a_comp_cor=args.a_comp_cor
use_aroma = args.use_aroma

# In[5]:

verboseprint('*'*79)
verboseprint(
    'Tasks: %s\n, Subjects: %s\n, derivatives_dir: %s\n, data_dir: %s' %
    (tasks, subjects, derivatives_dir, data_dir))
verboseprint('*'*79)

# In[6]:

to_run = []
for subject_id in subjects:
    for session in ses:
        for task in tasks:
            verboseprint('Setting up %s, %s' % (subject_id, task))
            files = get_first_level_objs(subject_id, session, task, first_level_dir, beta=False, regress_rt=regress_rt)
            if len(files) == 0 or args.overwrite:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore",category=DeprecationWarning)
                    warnings.filterwarnings("ignore",category=UserWarning)
                    subjinfo = make_first_level_obj(subject_id, session, task, fmriprep_dir, 
                                                    data_dir, first_level_dir, TR, regress_rt=regress_rt, a_comp_cor=a_comp_cor)
                if subjinfo is not None:
                    to_run.append(subjinfo)

# In[7]:

for subjinfo in to_run:
    verboseprint(subjinfo.ID)
    verboseprint('** fitting model')
    fmri_glm = FirstLevelModel(TR,
                               subject_label=subjinfo.ID,
                               mask=subjinfo.mask,
                               noise_model='ar1',
                               standardize=False,
                               hrf_model='spm',
                               drift_model='cosine',
                               n_jobs=1
                               )

    verboseprint('** saving')
    if not args.design_matrix_only:
        out = fmri_glm.fit(subjinfo.func, design_matrices=subjinfo.design)
        subjinfo.fit_model = out
    subjinfo.export_design(first_level_dir)
    subjinfo.export_events(first_level_dir)
    subjinfo.export_2ndlvl_meta(first_level_dir)
    save_first_level_obj(subjinfo, first_level_dir,
                         save_maps=(not args.design_matrix_only))
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
from glob import glob
from os import makedirs, path
import pandas as pd
import pickle
import sys

from nistats.second_level_model import SecondLevelModel
from nistats.thresholding import map_threshold
from nilearn import plotting
from utils.firstlevel_utils import (get_first_level_objs, 
                                    get_first_level_maps, 
                                    load_first_level_objs, 
                                    FirstLevel)
from utils.secondlevel_utils import create_group_mask, randomise
from utils.utils import get_contrasts, get_flags


# ### Parse Arguments
# These are not needed for the jupyter notebook, but are used after conversion to a script for production
# 
# - conversion command:
#   - jupyter nbconvert --to script --execute 2ndlevel_analysis.ipynb

# In[ ]:


parser = argparse.ArgumentParser(description='2nd level Entrypoint Script.')
parser.add_argument('-derivatives_dir', default=None)
parser.add_argument('--tasks', nargs="+", help="Choose from ANT, CCTHot, discountFix,                                     DPX, motorSelectiveStop, stopSignal,                                     stroop, surveyMedley, twoByTwo, WATT3")
parser.add_argument('--rerun', action='store_true')
parser.add_argument('--rt', action='store_true')
parser.add_argument('--beta', action='store_true')
parser.add_argument('--n_perms', default=1000, type=int)
parser.add_argument('--quiet', '-q', action='store_true')

if '-derivatives_dir' in sys.argv or '-h' in sys.argv:
    args = parser.parse_args()
else:
    args = parser.parse_args([])
    args.derivatives_dir = '/data/derivatives/'
    args.tasks = ['stroop']
    args.rt=True
    args.n_perms = 10


# In[ ]:


if not args.quiet:
    def verboseprint(*args, **kwargs):
        print(*args, **kwargs)
else:
    verboseprint = lambda *a, **k: None # do-nothing function


# ### Setup
# 
# Organize paths and set parameters based on arguments

# In[ ]:


# set paths
first_level_dir = path.join(args.derivatives_dir, '1stlevel')
second_level_dir = path.join(args.derivatives_dir,'2ndlevel')
fmriprep_dir = path.join(args.derivatives_dir, 'fmriprep')

# set tasks
if args.tasks is not None:
    tasks = args.tasks
else:
    tasks = ['ANT', 'CCTHot', 'discountFix',
            'DPX', 'motorSelectiveStop',
            'stopSignal', 'stroop',
            'twoByTwo', 'WATT3']
    
# set other variables
regress_rt = args.rt
beta_series = args.beta
n_perms = args.n_perms


# ### Create Mask

# In[ ]:


mask_threshold = .95
mask_loc = path.join(second_level_dir, 'group_mask_thresh-%s.nii.gz' % str(mask_threshold))
if path.exists(mask_loc) == False or args.rerun:
    verboseprint('Making group mask')
    group_mask = create_group_mask(fmriprep_dir, mask_threshold)
    makedirs(path.dirname(mask_loc), exist_ok=True)
    group_mask.to_filename(mask_loc)


# ### Create second level objects
# Gather first level models and create second level model

# In[ ]:


rt_flag, beta_flag = get_flags(regress_rt, beta_series)
for task in tasks:
    verboseprint('Running 2nd level for %s' % task)
    # load first level models
    # create contrast maps
    verboseprint('*** Creating maps')
    task_contrasts = get_contrasts(task, regress_rt)
    maps_dir = path.join(second_level_dir, task, 'secondlevel-%s_%s_maps' % (rt_flag, beta_flag))
    makedirs(maps_dir, exist_ok=True)
    # run through each contrast
    for name, contrast in task_contrasts:
        second_level_model = SecondLevelModel(mask=mask_loc, smoothing_fwhm=6)
        maps = get_first_level_maps('*', task, first_level_dir, name, regress_rt, beta_series)
        N = str(len(maps)).zfill(2)
        verboseprint('****** %s, %s files found' % (name, N))
        if len(maps) <= 1:
            verboseprint('****** No Maps')
            continue
        design_matrix = pd.DataFrame([1] * len(maps), columns=['intercept']) # ADD Age, Sex, RED score? make sure some are of interest and some are nuisance
        second_level_model.fit(maps, design_matrix=design_matrix)
        contrast_map = second_level_model.compute_contrast()
        # save
        contrast_file = path.join(maps_dir, 'contrast-%s.nii.gz' % name)
        contrast_map.to_filename(contrast_file)
         # write metadata
        with open(path.join(maps_dir, 'metadata.txt'), 'a') as f:
            f.write('Contrast-%s: %s maps\n' % (contrast, N))
        # save corrected map
        if n_perms > 0:
            verboseprint('*** Running Randomise')
            randomise(maps, maps_dir, mask_loc, n_perms=n_perms)
            # write metadata
            with open(path.join(maps_dir, 'metadata.txt'), 'a') as f:
                f.write('Contrast-%s: Randomise run with %s permutations\n' % (contrast, str(n_perms)))
    ##GROUP CONTRAST MAPS
    for group in ['BED', 'smoking']:
        verboseprint('*** Creating %s maps' % group)
        f = open("%s_subjects.txt" % group,"r") 
        group_subjects = f.read().split('\n')
        for name, contrast in task_contrasts:
            second_level_model = SecondLevelModel(mask=mask_loc, smoothing_fwhm=6)
            maps = []
            for curr_subject in group_subjects:
                maps.append(get_first_level_maps(curr_subject, task, first_level_dir, name, regress_rt, beta_series))
            N = str(len(maps)).zfill(2)
            verboseprint('****** %s, %s files found' % (name, N))
            if len(maps) <= 1:
                verboseprint('****** No Maps')
                continue
            design_matrix = pd.DataFrame([1] * len(maps), columns=['intercept'])
            second_level_model.fit(maps, design_matrix=design_matrix)
            contrast_map = second_level_model.compute_contrast()
            # save
            contrast_file = path.join(maps_dir, 'contrast-%s-%s.nii.gz' % (name, group))
            contrast_map.to_filename(contrast_file)
            # write metadata
            with open(path.join(maps_dir, 'metadata.txt'), 'a') as f:
                f.write('Contrast-%s-%s: %s maps\n' % (contrast, group, N))
            # save corrected map
            if n_perms > 0:
                verboseprint('*** Running Randomise')
                randomise(maps, maps_dir, mask_loc, n_perms=n_perms)
                # write metadata
                with open(path.join(maps_dir, 'metadata.txt'), 'a') as f:
                    f.write('Contrast-%s-%s: Randomise run with %s permutations\n' % (contrast, group, str(n_perms)))


    verboseprint('Done with %s' % task)


# In[ ]:


"""
# Using nistats method of first level objects. Not conducive for randomise.
rt_flag, beta_flag = get_flags(regress_rt, beta_series)
for task in tasks:
    verboseprint('Running 2nd level for %s' % task)
    # load first level models
    first_levels = load_first_level_objs(task, first_level_dir, regress_rt=regress_rt)
    if len(first_levels) == 0:
        continue
    first_level_models = [subj.fit_model for subj in first_levels]
    N = str(len(first_level_models)).zfill(2)

    # simple design for one sample test
    design_matrix = pd.DataFrame([1] * len(first_level_models), columns=['intercept'])
    
    # run second level
    verboseprint('*** Running model. %s first level files found' % N)
    second_level_model = SecondLevelModel(mask=mask_loc, smoothing_fwhm=6).fit(
        first_level_models, design_matrix=design_matrix)
    makedirs(path.join(second_level_dir, task), exist_ok=True)
    f = open(path.join(second_level_dir, task, 'secondlevel_%s_%s.pkl' % (rt_flag, beta_flag)), 'wb')
    pickle.dump(second_level_model, f)
    f.close()
    
    # create contrast maps
    verboseprint('*** Creating maps')
    task_contrasts = get_contrasts(task, regress_rt)
    maps_dir = path.join(second_level_dir, task, 'secondlevel_%s_%s_N-%s_maps' % (rt_flag, beta_flag, N))
    makedirs(maps_dir, exist_ok=True)
    for name, contrast in task_contrasts:
        verboseprint('****** %s' % name)
        contrast_map = second_level_model.compute_contrast(first_level_contrast=contrast)
        contrast_file = path.join(maps_dir, 'contrast-%s.nii.gz' % name)
        contrast_map.to_filename(contrast_file)
"""


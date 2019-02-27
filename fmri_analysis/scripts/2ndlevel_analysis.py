
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
from utils.firstlevel_utils import load_first_level_objs, FirstLevel
from utils.secondlevel_utils import create_group_mask
from utils.utils import get_contrasts


# ### Parse Arguments
# These are not needed for the jupyter notebook, but are used after conversion to a script for production
# 
# - conversion command:
#   - jupyter nbconvert --to script --execute 2ndlevel_analysis.ipynb

# In[ ]:


parser = argparse.ArgumentParser(description='fMRI Analysis Entrypoint Script.')
parser.add_argument('-derivatives_dir', default=None)
parser.add_argument('--tasks', nargs="+", help="Choose from ANT, CCTHot, discountFix,                                     DPX, motorSelectiveStop, stopSignal,                                     stroop, surveyMedley, twoByTwo, WATT3")
parser.add_argument('--rerun', action='store_true')
parser.add_argument('--rt', action='store_true')
parser.add_argument('--beta', action='store_true')

if '-derivatives_dir' in sys.argv or '-h' in sys.argv:
    args = parser.parse_args()
else:
    args = parser.parse_args([])
    args.derivatives_dir = '/data/derivatives'
    args.tasks = ['stroop']
    args.rt=True


# ### Setup

# In[ ]:


# set paths
first_level_dir = path.join(args.derivatives_dir, '1stlevel')
second_level_dir = path.join(args.derivatives_dir,'2ndlevel')
fmriprep_dir = path.join(args.derivatives_dir, 'fmriprep', 'fmriprep')

# set tasks
if args.tasks is not None:
    tasks = args.tasks
else:
    tasks = ['ANT', 'CCTHot', 'discountFix',
            'DPX', 'motorSelectiveStop',
            'stopSignal', 'stroop',
            'twoByTwo', 'WATT3']
    
# set other variables
subjects = sorted([i[-4:] for i in glob(path.join(first_level_dir, 's*'))])
regress_rt = args.rt
beta_series = args.beta


# ### Create Mask

# In[ ]:


mask_threshold = .95
mask_loc = path.join(second_level_dir, 'group_mask_thresh-%s.nii.gz' % str(mask_threshold))
if path.exists(mask_loc) == False or args.rerun:
    group_mask = create_group_mask(fmriprep_dir, mask_threshold)
    makedirs(path.dirname(mask_loc), exist_ok=True)
    group_mask.to_filename(mask_loc)


# ### Create second level objects

# In[ ]:


rt_flag = "True" if regress_rt else "False"
beta_flag = "True" if beta_series else "False"
for task in tasks:
    # load first level models
    first_levels = load_first_level_objs(subjects, task, first_level_dir, regress_rt=regress_rt)
    if len(first_levels) == 0:
        continue
    first_level_models = [subj.fit_model for subj in first_levels]
    # simple design for one sample test
    design_matrix = pd.DataFrame([1] * len(first_level_models), columns=['intercept'])
    # run second level
    second_level_model = SecondLevelModel(mask=mask_loc, smoothing_fwhm=4.4).fit(
        first_level_models, design_matrix=design_matrix)
    makedirs(path.join(second_level_dir, task), exist_ok=True)
    f = open(path.join(second_level_dir, task, 'secondlevel_RT-%s_beta-%s.pkl' % (rt_flag, beta_flag)), 'wb')
    pickle.dump(second_level_model, f)
    f.close()
    # create contrast maps
    task_contrasts = get_contrasts(task, regress_rt)
    maps_dir = path.join(second_level_dir, task, 'secondlevel_RT-%s_beta-%s_maps' % (rt_flag, beta_flag))
    makedirs(maps_dir, exist_ok=True)
    for name, contrast in task_contrasts:
        contrast_map = second_level_model.compute_contrast(first_level_contrast=contrast)
        zmaps[name] = contrast_map
        contrast_file = path.join(maps_dir, 'contrast-%s.nii.gz' % name)
        contrast_map.to_filename(contrast_file)


#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import argparse
from os import makedirs, path
import pandas as pd
import numpy as np
import sys

from nistats.second_level_model import SecondLevelModel
from utils.firstlevel_utils import get_first_level_maps
from utils.secondlevel_utils import create_group_mask, randomise
from utils.utils import get_contrasts, get_flags

# In[ ]:

parser = argparse.ArgumentParser(description='2nd level Entrypoint Script.')
parser.add_argument('-derivatives_dir', default=None)
parser.add_argument('--tasks', nargs="+", help="Choose from ANT, CCTHot, discountFix, DPX, motorSelectiveStop, stopSignal, stroop, surveyMedley, twoByTwo, WATT3")
parser.add_argument('--rerun', action='store_true')
parser.add_argument('--rt', action='store_true')
parser.add_argument('--beta', action='store_true')
parser.add_argument('--n_perms', default=1000, type=int)
parser.add_argument('--mask_thresh', default=.95, type=float)
parser.add_argument('--smoothing_fwhm', default=6)
parser.add_argument('--quiet', '-q', action='store_true')
parser.add_argument('--aim', default='NONE', help='Choose from aim1, aim2')
parser.add_argument('--group', default='NONE')

if '-derivatives_dir' in sys.argv or '-h' in sys.argv:
    args = parser.parse_args()
else:
    args = parser.parse_args([])
    args.derivatives_dir = '/data/derivatives/'
    args.tasks = ['stroop']
    args.rt = True
    args.n_perms = 10

# In[ ]:


if not args.quiet:
    def verboseprint(*args, **kwargs):
        print(*args, **kwargs)
else:
    def verboseprint(*args, **kwards):  # do-nothing function
        pass

# In[ ]:

# set paths
first_level_dir = path.join(args.derivatives_dir, '1stlevel')
second_level_dir = path.join(args.derivatives_dir, '2ndlevel')
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
group = args.group

# In[ ]:

# Create Mask
mask_loc = path.join(second_level_dir,
                     'group_mask_thresh-%s.nii.gz' % str(args.mask_thresh))
if (not path.exists(mask_loc)) or args.rerun:
    verboseprint('Making group mask at %s' % mask_loc)
    group_mask = create_group_mask(fmriprep_dir, args.mask_thresh)
    makedirs(path.dirname(mask_loc), exist_ok=True)
    group_mask.to_filename(mask_loc)

# In[ ]:


aim1_2ndlevel_confounds_path = "/scripts/aim1_2ndlevel_regressors/" +\
                               "aim1_2ndlevel_confounds_matrix.csv"
full_confounds_df = pd.read_csv(aim1_2ndlevel_confounds_path,
                                index_col='index')


def fit_and_compute_contrast(maps, task, second_level_model):
    design_matrix, curr_contrasts = get_group_DM_and_contrasts(maps,
                                                               task)
    maps, design_matrix = filter_maps_and_DM(maps, design_matrix)
    second_level_model.fit(maps,
                           design_matrix=design_matrix)
    contrast_map = second_level_model.compute_contrast(
        second_level_contrast=curr_contrasts)
    return contrast_map, maps


def get_group_DM_and_contrasts(maps, task):
    design_matrix = pd.DataFrame([1] * len(maps), columns=['intercept'])
    if args.aim == 'aim1':
        subjects = [m.split('1stlevel/')[-1].split('/')[0] for m in maps]
        design_matrix = full_confounds_df.loc[subjects,
                                              ['age', 'sex', task+'_meanFD']
                                              ].copy()
        design_matrix.index.rename('subject_label', inplace=True)
        design_matrix['intercept'] = 1
    if args.aim == 'aim1_noFD':
        subjects = [m.split('1stlevel/')[-1].split('/')[0] for m in maps]
        design_matrix = full_confounds_df.loc[subjects, ['age', 'sex']].copy()
        design_matrix.index.rename('subject_label', inplace=True)
        design_matrix['intercept'] = 1
    ncols = design_matrix.shape[1]
    contrasts = np.zeros(ncols)
    contrasts[-1] = 1
    contrasts = [int(i) for i in contrasts]
    print(contrasts)
    print(design_matrix.head())
    return design_matrix, contrasts


def filter_maps_and_DM(maps, design_matrix):
    drop_num = design_matrix.isna().any(axis=1).sum()
    print('dropping ' +
          str(drop_num) +
          ' due to missing values in design matrix')
    design_matrix = design_matrix.dropna()
    if len(design_matrix) != len(maps):
        keep_subs = design_matrix.index.tolist()
        maps = [m for m in maps if
                m.split('1stlevel/')[-1].split('/')[0] in keep_subs]
    assert(len(design_matrix) == len(maps))
    return maps, design_matrix


rt_flag, beta_flag = get_flags(regress_rt, beta_series)
for task in tasks:
    verboseprint('Running 2nd level for %s' % task)

    verboseprint('*** Creating maps')
    task_contrasts = get_contrasts(task, regress_rt)
    maps_dir = path.join(second_level_dir,
                         task,
                         'secondlevel-%s_%s_maps' % (rt_flag, beta_flag))
    makedirs(maps_dir, exist_ok=True)

    # run through each contrast for all participants
    if group == 'NONE':
        for name, contrast in task_contrasts:
            second_level_model = SecondLevelModel(
                mask=mask_loc,
                smoothing_fwhm=args.smoothing_fwhm
                )
            maps = get_first_level_maps('*', task,
                                        first_level_dir,
                                        name,
                                        regress_rt,
                                        beta_series)
            N = str(len(maps)).zfill(2)
            verboseprint('****** %s, %s files found' % (name, N))
            if len(maps) <= 1:
                verboseprint('****** No Maps')
                continue

            contrast_map, maps = fit_and_compute_contrast(maps,
                                                          task,
                                                          second_level_model)
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
                    f.write(
                        'Contrast-%s: Randomise run with %s permutations\n' %
                        (contrast, str(n_perms)))

    else:
        verboseprint('*** Creating %s maps' % group)
        f = open("/scripts/%s_subjects.txt" % group, "r")
        group_subjects = f.read().split('\n')
        for name, contrast in task_contrasts:
            second_level_model = SecondLevelModel(
                mask=mask_loc,
                smoothing_fwhm=args.smoothing_fwhm
                )
            maps = []
            for curr_subject in group_subjects:
                curr_map = get_first_level_maps(curr_subject, task,
                                                first_level_dir,
                                                name,
                                                regress_rt,
                                                beta_series)
                if len(curr_map):
                    maps += curr_map
            N = str(len(maps)).zfill(2)
            verboseprint('****** %s, %s files found' % (name, N))
            if len(maps) <= 1:
                verboseprint('****** No Maps')
                continue
            contrast_map, maps = fit_and_compute_contrast(maps,
                                                          task,
                                                          second_level_model)
            # save
            group_dir = path.join(maps_dir, group)
            makedirs(group_dir, exist_ok=True)
            contrast_file = path.join(group_dir,
                                      'contrast-%s-%s.nii.gz' % (name, group))
            contrast_map.to_filename(contrast_file)
            # write metadata
            with open(path.join(group_dir, 'metadata.txt'), 'a') as f:
                f.write('Contrast-%s-%s: %s maps\n' % (contrast, group, N))
            # save corrected map
            if n_perms > 0:
                verboseprint('*** Running Randomise')
                randomise(maps, group_dir,
                          mask_loc,
                          n_perms=n_perms,
                          group=group)
                # write metadata
                with open(path.join(group_dir, 'metadata.txt'), 'a') as f:
                    f.write(
                        'Contrast-%s-%s: Randomise run with %s permutations\n'
                        %
                        (contrast, group, str(n_perms)))

    verboseprint('Done with %s' % task)

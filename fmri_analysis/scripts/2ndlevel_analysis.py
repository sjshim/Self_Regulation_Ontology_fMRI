#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import argparse
from os import makedirs, path
import pandas as pd
import numpy as np
import sys
import json

from nistats.second_level_model import SecondLevelModel
from utils.firstlevel_utils import get_first_level_maps, get_first_level_metas
from utils.secondlevel_utils import create_group_mask, randomise
from utils.utils import get_contrasts, get_flags

# In[ ]:
def get_args():
    parser = argparse.ArgumentParser(description='2nd level Entrypoint Script.')
    parser.add_argument('-derivatives_dir', default=None)
    parser.add_argument('--tasks', nargs="+", help="Choose from ANT, CCTHot, discountFix, DPX, motorSelectiveStop, stopSignal, stroop, surveyMedley, twoByTwo, WATT3")
    parser.add_argument('--rerun', action='store_true')
    parser.add_argument('--rt', action='store_true')
    parser.add_argument('--beta', action='store_true')
    parser.add_argument('--n_perms', default=10000, type=int)
    parser.add_argument('--mask_thresh', default=.95, type=float)
    parser.add_argument('--smoothing_fwhm', default=6)
    parser.add_argument('--scnd_lvl', default='NONE')
    parser.add_argument('--quiet', '-q', action='store_true')
    parser.add_argument('--aim', default='NONE', help='Choose from aim1, aim2')
    parser.add_argument('--group', default='NONE')

    if '-derivatives_dir' in sys.argv or '-h' in sys.argv:
        args = parser.parse_args()
    else:
        args = parser.parse_args([])
        args.derivatives_dir = '/data/derivatives/'
        args.tasks = ['stroop']
        args.scnd_lvl = 'intercept'
        args.rt = True
        args.n_perms = 10
    return args

def extend_confounds_df(indiv_meta_files):
    aim1_2ndlevel_confounds_path = "/scripts/aim1_2ndlevel_regressors/" +\
                               "aim1_2ndlevel_confounds_matrix.csv"
    full_confounds_df = pd.read_csv(aim1_2ndlevel_confounds_path,
                                index_col='index')
    meta_dict = {}
    for meta_file in indiv_meta_files:
        sub_id = meta_file.replace(first_level_dir+'/', '').split('/%s' % task)[0]
        with open(meta_file, 'r') as f:
            meta_dict[sub_id] = json.load(f)
    meta_df = pd.DataFrame(meta_dict).T

    extended_confounds_df = pd.concat([meta_df, full_confounds_df], axis=1, sort=True).copy()
    # substitute in old mriqc if new is not available
    extended_confounds_df['FD_mean'] =  extended_confounds_df['FD_mean'].combine_first(extended_confounds_df['%s_meanFD' % task])
    return extended_confounds_df


def get_2ndlevel_desMat(maps, task, extended_confounds_df):
    subjects = [m.split('1stlevel/')[-1].split('/')[0] for m in maps]
    rt_cols = extended_confounds_df.filter(regex='RT').columns
    dm_cols = ['age', 'sex'] + list(rt_cols) + ['FD_mean']
    des_mat = extended_confounds_df.loc[subjects,
                                            dm_cols,
                                            ].copy()
    des_mat.index.rename('subject_label', inplace=True)
    des_mat.insert(0, 'intercept', 1)
    demean_cols = [col for col in des_mat.columns if col!='intercept']
    for col in demean_cols: #demean to continue capturing mean effect.
        des_mat[col] -= des_mat[col].mean()
    return des_mat

def filter_maps_and_DM(maps, des_mat):
    drop_num = des_mat.isna().any(axis=1).sum()
    print('dropping ' +
          str(drop_num) +
          ' due to missing values in design matrix')
    des_mat = des_mat.dropna()
    if len(des_mat) != len(maps):
        keep_subs = des_mat.index.tolist()
        maps = [m for m in maps if
                m.split('1stlevel/')[-1].split('/')[0] in keep_subs]
    assert(len(des_mat) == len(maps))
    return maps, des_mat

def get_2ndlevel_contrast(des_mat, scnd_lvl):
    ncols = des_mat.shape[1]
    contrast = np.zeros(ncols)
    contrast[des_mat.columns.get_loc(scnd_lvl)] = 1
    contrast = [int(i) for i in contrast]
    return contrast

if __name__=='__main__':
    args = get_args()

    if not args.quiet:
        def verboseprint(*args, **kwargs):
            print(*args, **kwargs)
    else:
        def verboseprint(*args, **kwards):  # do-nothing function
            pass

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

    # Create Mask
    mask_loc = path.join(second_level_dir,
                        'group_mask_thresh-%s.nii.gz' % str(args.mask_thresh))
    if (not path.exists(mask_loc)) or args.rerun:
        verboseprint('Making group mask at %s' % mask_loc)
        group_mask = create_group_mask(fmriprep_dir, args.mask_thresh)
        makedirs(path.dirname(mask_loc), exist_ok=True)
        group_mask.to_filename(mask_loc)

    rt_flag, beta_flag = get_flags(regress_rt, beta_series)
    for task in tasks:
        verboseprint('Running 2nd level for %s' % task)

        verboseprint('*** Creating maps')
        task_contrasts = get_contrasts(task, regress_rt)
        maps_dir = path.join(second_level_dir,
                            task,
                            'secondlevel-%s_%s_maps' % (rt_flag, beta_flag))
        makedirs(maps_dir, exist_ok=True)

        indiv_meta_files = get_first_level_metas('*', task,
                                                first_level_dir,
                                                regress_rt,
                                                beta_series)
        extended_confounds_df = extend_confounds_df(indiv_meta_files)

        # run through each contrast for all participants
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

            # run 2ndlevels
            des_mat = get_2ndlevel_desMat(maps,
                                        task,
                                        extended_confounds_df)
            maps, des_mat = filter_maps_and_DM(maps, des_mat)
            second_level_model.fit(maps, design_matrix=des_mat)
            second_level_contrast = get_2ndlevel_contrast(des_mat, args.scnd_lvl)
            contrast_map = second_level_model.compute_contrast(
                second_level_contrast=second_level_contrast
                )
            # save
            contrast_file = path.join(maps_dir, 'contrast-%s_2ndlevel-%s.nii.gz' % (name, args.scnd_lvl))
            contrast_map.to_filename(contrast_file)
            # write metadata
            N = str(len(maps)).zfill(2)
            with open(path.join(maps_dir, 'metadata.txt'), 'a') as f:
                f.write('Contrast-%s: %s maps\n' % (contrast, N))
            # save corrected map
            if n_perms > 0:
                verboseprint('*** Running Randomise')
                randomise(maps, maps_dir, mask_loc, des_mat, args.scnd_lvl,
                          fwhm=args.smoothing_fwhm,
                          n_perms=n_perms)
                # write metadata
                with open(path.join(maps_dir, 'metadata.txt'), 'a') as f:
                    f.write(
                        'Contrast-%s: Randomise run with %s permutations\n' %
                        (contrast, str(n_perms)))

        verboseprint('Done with %s' % task)

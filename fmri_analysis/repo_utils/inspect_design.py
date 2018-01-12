#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from glob import glob
from os.path import join
import pandas as pd
import pickle
from utils.display_utils import get_design_df, plot_design

data_dir = '/mnt/OAK/fmri_analysis/output/1stLevel/'
task = 'ANT'

# get design/events files that have been output from nipype
subj_index = 0
paths = sorted(glob(join(data_dir, '*%s*' % task)))
path = paths[subj_index]
df = get_design_df(path)

contrast_paths = sorted(glob(join(data_dir, '*%s*' % task, 'contrasts.pkl')))
contrast_path = contrast_paths[subj_index]
contrasts = pickle.load(open(contrast_path, 'rb'))

# get events_file
event_paths = sorted(glob(join('../../Data/event_files','*')))
event_path = event_paths[subj_index]
events_df = pd.DataFrame.from_csv(event_path, sep='\t')

# get individual contrast files that have been output from nipype

from nilearn.image import concat_imgs, mean_img, threshold_img
from nilearn.plotting import plot_glass_brain, plot_stat_map

for task in ['ANT', 'stroop']:
    contrast_paths = sorted(glob(join(data_dir, '*%s*' % task, 'contrasts.pkl')))
    contrast_path = contrast_paths[subj_index]
    contrasts = pickle.load(open(contrast_path, 'rb'))

    for contrast_num in range(1, len(contrasts)+1):
        contrast_name = contrasts[contrast_num-1][0]
        paths = sorted(glob(join(data_dir, '*%s*' % task, '*tstat%s*' % contrast_num)))
        imgs = concat_imgs(paths)
        thresh_imgs = threshold_img(imgs, 2)
        contrast_percent = mean_img(thresh_imgs)
        plot_stat_map(contrast_percent, title=contrast_name,
                      output_file = '../Plots/%s_%s_percent_exceed.png' % (task, contrast_name),
                      vmax=1)

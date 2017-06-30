#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 11:17:08 2017

@author: ian
"""
from matplotlib import pyplot as plt
import nilearn.image
import nilearn.plotting
import sys
sys.path.append('../Docker/scripts/utils')
from display_utils import get_design_df, plot_design

task = 'ANT'

fig, axes = plt.subplots(4, 1, figsize=(24, 15), squeeze=True)

img = '/mnt/Sherlock_Scratch/output/1stLevel/s546_task_stroop/cope3.nii.gz'
smooth_cope_rt = nilearn.image.smooth_img(img, 8)
nilearn.plotting.plot_glass_brain(smooth_cope_rt,
                                  display_mode='lyrz', 
                                  colorbar=True, 
                                  plot_abs=False,
                                  title='RT',
                                  axes = axes[0])


img = '/mnt/Sherlock_Scratch/output_noRT/1stLevel/s546_task_stroop/cope3.nii.gz'
smooth_cope_noRT = nilearn.image.smooth_img(img, 8)
nilearn.plotting.plot_glass_brain(smooth_cope_noRT,
                                  display_mode='lyrz', 
                                  colorbar=True, 
                                  plot_abs=False,
                                  title='noRT',
                                  axes = axes[1])


diff_img = nilearn.image.math_img("img1 - img2", 
                                    img1=smooth_cope_rt, 
                                    img2=smooth_cope_noRT)

nilearn.plotting.plot_glass_brain(diff_img,
                                  display_mode='lyrz', 
                                  colorbar=True, 
                                  plot_abs=False,
                                  title='diff',
                                  axes = axes[2])

img = '/mnt/Sherlock_Scratch/output/1stLevel/s546_task_stroop/cope4.nii.gz'
smooth_cope_rtmap = nilearn.image.smooth_img(img, 8)
nilearn.plotting.plot_glass_brain(smooth_cope_rtmap,
                                  display_mode='lyrz', 
                                  colorbar=True, 
                                  plot_abs=False,
                                  title='RT_map',
                                  axes = axes[3])


df = get_design_df('/mnt/Sherlock_Scratch/output/1stLevel/s546_task_%s' % task)
df_noRT = get_design_df('/mnt/Sherlock_Scratch/output_noRT/1stLevel/s546_task_%s' % task)

plot_design(df)

# coding: utf-8

# In[ ]:


import argparse
from glob import glob
import json
from matplotlib import pyplot as plt
from nilearn import datasets, image
from nilearn import plotting
from nilearn.image import iter_img
from os import makedirs, path
import pandas as pd
import seaborn as sns
import sys

from utils.secondlevel_plot_utils import *


# In[ ]:


parser = argparse.ArgumentParser(description='fMRI Analysis Entrypoint Script.')
parser.add_argument('-derivatives_dir')
parser.add_argument('--tasks', nargs="+")

if '-derivatives_dir' in sys.argv or '-h' in sys.argv:
    matplotlib.use("agg")
    args = parser.parse_args()
else:
    # if run as a notebook reduce the set of args
    args = parser.parse_args([])
    args.derivatives_dir='/data/derivatives'


# In[ ]:


derivatives_dir = args.derivatives_dir
fmriprep_dir = path.join(derivatives_dir, 'fmriprep', 'fmriprep')
first_level_dir = path.join(derivatives_dir, '1stlevel')
second_level_dir = path.join(derivatives_dir,'2ndlevel')  
tasks = ['ANT', 'CCTHot', 'discountFix',
         'DPX', 'motorSelectiveStop',
         'stopSignal', 'stroop', 
         'surveyMedley', 'twoByTwo', 'WATT3']
if args.tasks:
    tasks = args.tasks


# In[ ]:


task = 'stroop'
for model in ['model-rt', 'model-nort']:
    for task in tasks:
        task_contrast_dirs = sorted(glob(path.join(second_level_dir, '*%s' % task, model, 'wf-contrast')))
        for d in task_contrast_dirs:
            save_loc = path.join(d, 'Plots')
            plot_2ndlevel_maps(d, threshold=.95, plot_dir=save_loc, ext='png')

print('Plotting task contrasts...')
for task in tasks:
    for tfile in ['raw', 'correct']:
        task_dir = path.join(data_dir, task)
        subj_ids = json.load(open(path.join(task_dir,'subj_ids.json'),'r'))
        tstat_files = sorted(glob(path.join(task_dir, '*%s*%s_tfile*' % (task, tfile))),
                             key = lambda x: '-' in x)
        group_fig, group_axes = plt.subplots(len(tstat_files), 1,
                                         figsize=(14, 6*len(tstat_files)))
        group_fig.suptitle('N = %s' % len(subj_ids), fontsize=30)
        plt.subplots_adjust(top=.95)
        for i, tfile in enumerate(tstat_files):
            basename = path.basename(tfile)
            title = basename[:(basename.find('raw')-1)]
            plotting.plot_stat_map(tfile, threshold=2, 
                                   axes=group_axes[i],
                                   title=title)
        makedirs(path.join(output_dir,task), exist_ok=True)
        group_fig.savefig(path.join(output_dir,task,'%s_%s_tfiles.png' % (task, tfile)))
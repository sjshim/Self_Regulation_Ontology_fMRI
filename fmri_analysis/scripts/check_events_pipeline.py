import argparse
from glob import glob
from os import path
import pandas as pd
import seaborn as sns
import sys
sys.path.append('../Docker/scripts/')
from utils.display_utils import get_design_df, plot_design
parser = argparse.ArgumentParser()
parser.add_argument('fmri_dir', help='The directory where both the fmriprep and' + \
                                      '1st level analysis can be found')
parser.add_argument('--task')

args = parser.parse_args()
fmri_dir = args.fmri_dir
task= args.task

subj = 's525'

print('*'*79)
print('%s: %s' % (subj, task))
print('*'*79)

cleaned_dir = '../../Data/processed/%s_%s_cleaned.csv' % (subj, task)
cleaned_file = pd.read_csv(cleaned_dir)

events_dir = path.join(fmri_dir,'fmriprep/fmriprep/sub-%s/ses-*/func' % subj)
events_dir = glob(path.join(events_dir, '*%s*events.tsv' % task))[0]
events_file = pd.read_csv(events_dir, sep='\t')

task_dir = path.join(fmri_dir, 'output/1stLevel/%s_task_%s/' % (subj, task))
design = get_design_df(task_dir)
plot_design(design)


sns.set_context('notebook', font_scale=1.4)
rt_index = list(design.columns).index('response_time')
cols = design.columns[:rt_index+1:2]
sns.plt.figure(figsize=(16,8*len(cols)))
for i, col in enumerate(cols):
    sns.plt.subplot(len(cols), 1, i+1)
    if col in events_file.columns:
        sns.plt.plot(events_file.onset, events_file[col], label='event_%s' % col)
    sns.plt.plot(design.index*.68, design[col], label='design_%s' % col)
    sns.plt.legend()
    sns.plt.title(col)








from utils.utils import process_confounds, parse_EVs

data_dir = path.join(fmri_dir, 'fmriprep', 'fmriprep')
confounds_file = glob(path.join(data_dir,
                               'sub-%s' % subj,
                               '*', 'func',
                               '*%s*confounds.tsv' % task))[0]
regressors, regressor_names = process_confounds(confounds_file)

conditions, onsets, durations, amplitudes = parse_EVs(events_file, 
                                                          task,
                                                          regress_rt=True)
import argparse
from inspect import currentframe, getframeinfo
import nilearn
from pathlib import Path
from os import path
from utils.plot_utils import get_design_df, plot_design, plot_fmri_resid, plot_1stlevel_maps
from utils.utils import get_event_dfs, load_atlas
"""
# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('derivatives_dir')
parser.add_argument('data_dir')
parser.add_argument('--participant_labels', nargs="+")
parser.add_argument('--tasks', nargs="+")
args = parser.parse_args()

derivatives_dir = args.derivatives_dir
data_dir = args.data_dir
# list of subject identifiers
subject_list = args.participant_labels
# list of task identifiers
if args.tasks:
    task_list = args.tasks
else:
  task_list = ['ANT', 'CCTHot', 'discountFix',
               'DPX', 'motorSelectiveStop',
               'stopSignal', 'stroop', 'surveyMedley',
               'twoByTwo', 'WATT3']
"""

derivatives_dir = '/mnt/OAK/derivatives/'
data_dir = '/mnt/OAK'
task_list = ['stroop']

filename = getframeinfo(currentframe()).filename
current_directory = str(Path(filename).resolve().parent)

# set up atlas
# glasser atlas
atlas_path = path.join(current_directory, 'atlases', 'HCPMMP1_on_MNI152_ICBM2009a_nlin_hd.nii.gz')
atlas_label_path = path.join(current_directory, 'atlases', 'HCPMMP1_on_MNI152_ICBM2009a_nlin.txt')
try:
    atlas = load_atlas(atlas_path, atlas_label_path)
except FileNotFoundError:
    print('Glasser atlas not found')
# harvard atlas
atlas=nilearn.datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')


task = 'surveyMedley'
subject = 's582'
model = 'model-rt'
for wf in ['wf-beta']:
    # inspect design
    GLM_path = path.join(derivatives_dir, '1stlevel', subject, task, model, wf)
    design_df = get_design_df(GLM_path)
    plot_design(design_df, size=20)
    contrast_files = plot_1stlevel_maps(GLM_path)

    # inspect event
    event_df = get_event_dfs(data_dir, subj=subject, task=task)[subject][task]
    
    # inspect residuals
    resid_path = path.join(derivatives_dir, '1stLevel', subject, task, model, wf, 'res4d.nii.gz')
    time_series = plot_fmri_resid(resid_path, atlas)

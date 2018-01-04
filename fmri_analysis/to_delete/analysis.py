#!/usr/bin/env python3
import argparse
import os
import subprocess
import nibabel
import numpy
from glob import glob

parser = argparse.ArgumentParser(description='fMRI Analysis Entrypoint Script.')
parser.add_argument('output_dir', help='The directory where the output files '
                    'should be stored. If you are running group level analysis '
                    'this folder should be prepopulated with the results of the'
                    'participant level analysis.')
parser.add_argument('--data_dir')
parser.add_argument('--mask_dir')
parser.add_argument('--participant_label', help='The label(s) of the participant(s) that should be analyzed. The label '
                   'corresponds to sub-<participant_label> from the BIDS spec '
                   '(so it does not include "sub-").',
                   nargs="+")
parser.add_argument('--tasks',help='The label(s) of the task(s)'
                   'that should be analyzed. If this parameter is not '
                   'provided all tasks should be analyzed.',
                   nargs="+")
parser.add_argument('--ignore_rt', action='store_true', 
                    help='Bool, defaults to True. If true include response'
                    'time as a regressor')
parser.add_argument('--script', default='task_analysis.py',
                    help='Script to run. Options: task_analysis.py (default), group_plots.py, group_analysis.py')
parser.add_argument('--cleanup', action='store_true', 
                    help='If included, delete working directory')


args, unknown = parser.parse_known_args()
output_dir = args.output_dir
script = args.script
assert os.path.exists(output_dir), \
  "Output directory %s doesn't exist!" % output_dir
  
cmd = "python /scripts/%s " % script + output_dir

# participant and task options
if args.participant_label:
    cmd += " --participant_label " + ' '.join(args.participant_label)

if args.tasks:
    task_list = args.tasks
    cmd += ' --tasks ' + ' '.join(task_list)
    
# directories
if args.data_dir:
    assert os.path.exists(args.data_dir), \
        "Data directory %s doesn't exist!" % args.data_dir
    cmd += " --data_dir " + args.data_dir

if args.mask_dir:
    assert os.path.exists(args.mask_dir), \
        "Mask directory %s doesn't exist!" % args.mask_dir
    cmd += " --mask_dir " + args.mask_dir

# other options
if args.ignore_rt:
    cmd += ' --ignore_rt'

if args.cleanup:
    cmd += ' --cleanup'

print('*'*79)
print('Executing Command: %s' % cmd)
print('*'*79)

# run fmri_analysis
os.system(cmd)
    

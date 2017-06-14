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
parser.add_argument('--data_dir',help='The label(s) of the participant(s)'
                   'that should be analyzed. Multiple '
                   'participants can be specified with a space separated list.')
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


args = parser.parse_args()
output_dir = args.output_dir
assert os.path.exists(output_dir), \
  "Output directory %s doesn't exist!" % output_dir
# subset of subjects
subjects_to_analyze = args.participant_label
cmd = "python /scripts/task_analysis.py " + \
      output_dir + \
      " --participant_label " + ' '.join(subjects_to_analyze)

if args.data_dir:
    assert os.path.exists(args.data_dir), \
        "Data directory %s doesn't exist!" % args.data_dir
    cmd += " --data_dir " + args.data_dir

if args.tasks:
    task_list = args.tasks
    cmd += ' --task ' + ' '.join(task_list)
    
if args.ignore_rt:
    cmd += ' --ignore_rt'
# run fmri_analysis
os.system(cmd)
    

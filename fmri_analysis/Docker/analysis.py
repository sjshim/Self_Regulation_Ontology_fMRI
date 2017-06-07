#!/usr/bin/env python3
import argparse
import os
import subprocess
import nibabel
import numpy
from glob import glob

parser = argparse.ArgumentParser(description='fMRI Analysis Entrypoint Script.')
parser.add_argument('--participant_label', help='The label(s) of the participant(s) that should be analyzed. The label '
                   'corresponds to sub-<participant_label> from the BIDS spec '
                   '(so it does not include "sub-").',
                   nargs="+")
parser.add_argument('--tasks',help='The label(s) of the task(s)'
                   'that should be analyzed. If this parameter is not '
                   'provided all tasks should be analyzed.',
                   nargs="+")


args = parser.parse_args()

# subset of subjects
subjects_to_analyze = args.participant_label
cmd = "python /scripts/task_analysis.py --participant_label " + ' '.join(subjects_to_analyze)

if args.tasks:
    task_list = args.tasks
    cmd += ' --task ' + ' '.join(task_list)
# run fmri_analysis
os.system(cmd)
    

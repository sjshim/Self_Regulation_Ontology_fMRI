#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import argparse
from glob import glob
from os import path
import sys

from nilearn.decomposition import CanICA


# In[ ]:

parser = argparse.ArgumentParser(description='First Level Inspection Entrypoint script')
parser.add_argument('-derivatives_dir', default=None)
parser.add_argument('--tasks', nargs="+", help="Choose from ANT, CCTHot, discountFix,                                     DPX, motorSelectiveStop, stopSignal,                                     stroop, surveyMedley, twoByTwo, WATT3")
parser.add_argument('-n_procs', default=1, type=int)
if '-derivatives_dir' in sys.argv or '-h' in sys.argv:
    args = parser.parse_args()
else:
    args = parser.parse_args([])
    args.derivatives_dir = '/data/derivatives'
    args.n_procs=1
    args.tasks = ['stopSignal']


# In[ ]:

fmriprep_dir = path.join(args.derivatives_dir, 'fmriprep', 'fmriprep')
first_level_dir = path.join(args.derivatives_dir,'1stlevel')
# set tasks
if args.tasks is not None:
    tasks = args.tasks
else:
    tasks = ['ANT', 'CCTHot', 'discountFix',
            'DPX', 'motorSelectiveStop',
            'stopSignal', 'stroop',
            'twoByTwo', 'WATT3']
n_comps = 20


# # Run Canonical ICA

# In[ ]:

for task in tasks:
    func_filenames = glob(path.join(fmriprep_dir, '*', '*', 'func', '*%s*MNI*preproc.nii.gz' % task))
    canica = CanICA(n_components=n_comps, smoothing_fwhm=6.,
                    threshold=3., verbose=10, random_state=0,
                    n_jobs=args.n_procs)
    canica.fit(func_filenames)
    components_img = canica.components_img_
    components_img.to_filename(path.join(first_level_dir, '%s_canica_NComp-%s.nii.gz' % (task, str(n_comps)))


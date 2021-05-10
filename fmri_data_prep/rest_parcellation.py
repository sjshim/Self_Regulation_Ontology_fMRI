from nilearn import image
from glob import glob
import pandas as pd
from nilearn.input_data import NiftiLabelsMasker
from templateflow import api as tflow
import argparse
import sys 
import os


parser = argparse.ArgumentParser(description='First Level Entrypoint script')
parser.add_argument('-bids_dir', default='/data')
parser.add_argument('-atlas', default=None)
parser.add_argument('-output_dir', default='/data/derivatives/parcellations/rest')
parser.add_argument('-atlas_dir', default='./')

if '-bids_dir' in sys.argv:
    args = parser.parse_args()
else:
    args = parser.parse_args([])
    oak_mount = '/Users/henrymj/Documents/mounts/OAK'
    args.bids_dir = os.path.join(oak_mount, 'data/uh2/aim1/BIDS_scans')
    args.atlas = 400
    args.output_dir = os.path.join(args.bids_dir, 'derivatives/parcellations/rest')

bids_dir = args.bids_dir
output_dir = args.output_dir

first_level_dir  = os.path.join(bids_dir,'derivatives', '1stlevel')   

if args.atlas=='combo':
    atlas_path = os.path.join(args.atlas_dir, 'Parcels_Combo.nii.gz')
elif args.atlas=='SUIT':
    atlas_path = os.path.join(args.atlas_dir, 'SUIT.nii.gz')
else:  # assuming shaefer atlas for now
    spec_dict = {'atlas': 'Schaefer2018',
                 'desc': f'{args.atlas}Parcels17Networks',
                 'nparcels': args.atlas,
                 'nnetworks': 17,
                 'atlas': 'Schaefer2018',
                 'atlas_resolution': 2,
                 'space': 'MNI152NLin2009cAsym',
                 'atlas_desc': f'{args.atlas}Parcels17Networks'}

    atlas_path =  tflow.get(spec_dict['space'],
                  desc=spec_dict['atlas_desc'],
                  resolution=spec_dict['atlas_resolution'],
                  atlas=spec_dict['atlas'])
    atlas_path = str(atlas_path)

masker = NiftiLabelsMasker(labels_img=os.path.abspath(atlas_path), 
                           standardize=True,
                           memory='nilearn_cache',
                           verbose=5)


for rest_file in glob(os.path.join(first_level_dir,
                                        '*',  # sub
                                        'rest',  # task
                                        'ses-*'
                                        '*.nii.gz')
                                        ):
    subid = rest_file.split('1stlevel/')[1].split('/')[0]
    ses = rest_file.split('ses-')[1].split('/')[0]  # sub/rest/ses-2/...
    run = rest_file.split('run-')[-1].split('_')[0]
    beta_array = masker.fit_transform(rest_file)
    pd.DataFrame(beta_array).to_csv(os.path.join(args.output_dir,
            f'{subid}_task-rest_ses-{ses}_run-{run}_atlas-{args.atlas}.csv'))

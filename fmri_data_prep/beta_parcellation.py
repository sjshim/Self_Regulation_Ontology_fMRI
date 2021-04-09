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
parser.add_argument('-output_dir', default='/data/derivatives/parcellations/1stlevel_beta')
parser.add_argument('-atlas_dir', default='./')
parser.add_argument('--tasks', nargs="+", help="Choose from ANT, CCTHot, discountFix, \
            DPX, motorSelectiveStop, stopSignal, stroop, surveyMedley, twoByTwo, WATT3")
parser.add_argument('--RT_flag', nargs='+', default=['RT-True', 'RT-False'], help='Choose from RT-True, RT-False')

if '-bids_dir' in sys.argv:
    args = parser.parse_args()
else:
    args = parser.parse_args([])
    oak_mount = '/Users/henrymj/Documents/mounts/OAK'
    args.bids_dir = os.path.join(oak_mount, 'data/uh2/aim1/BIDS_scans')
    args.atlas = 400
    args.output_dir = os.path.join(args.bids_dir, 'derivatives/parcellations/1stlevel_beta')


tasks = ['ANT', 'CCTHot', 'DPX', 'discountFix', 'motorSelectiveStop', 
                 'stopSignal', 'stroop', 'twoByTwo', 'WATT3']
bids_dir = args.bids_dir
output_dir = args.output_dir

first_level_dir  = os.path.join(bids_dir,'derivatives', '1stlevel')   

if args.atlas=='combo':
    atlas_path = os.path.join(atlas_dir, 'Parcels_Combo.nii.gz')
elif args.atlas=='SUIT':
    atlas_path = os.path.join(atlas_dir, 'SUIT.nii.gz')
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

for RT_flag in args.RT_flag:
    curr_output_dir = os.path.join(output_dir, RT_flag)
    os.makedirs(curr_output_dir, exist_ok=True)
    for contrast_file in glob(os.path.join(first_level_dir,
                                           '*',  # sub
                                           '*',  # task
                                           f'maps_{RT_flag}_beta-False',
                                           'contrast-*.nii.gz')
                                           ):
        subid = contrast_file.split('1stlevel/')[1].split('/')[0]
        task = contrast_file.split(subid+'/')[1].split('/')[0]
        beta_array = masker.fit_transform([contrast_file])
        contrast = contrast_file.split('contrast-')[-1].replace('.nii.gz', '')
        pd.DataFrame(beta_array).to_csv(os.path.join(curr_output_dir,
                f'{subid}_task-{task}_contrast-{contrast}_atlas-{args.atlas}.csv'))

from nilearn import image
from glob import glob
import pandas as pd
from nilearn.input_data import NiftiLabelsMasker
import argparse
import sys 
import os


parser = argparse.ArgumentParser(description='First Level Entrypoint script')
parser.add_argument('-bids_dir', default='/data')
parser.add_argument('-parcellation_loc', default='./Parcels_Combo.nii.gz')
parser.add_argument('-output_dir', default='/data/derivatives/1stlevel/beta_parcellations')
parser.add_argument('--tasks', nargs="+", help="Choose from ANT, CCTHot, discountFix, \
            DPX, motorSelectiveStop, stopSignal, stroop, surveyMedley, twoByTwo, WATT3")

if '-bids_dir' in sys.argv or '-h' in sys.argv:
    args = parser.parse_args()
else:
    args = parser.parse_args([])
    args.bids_dir = 'tmp/OAK/data/uh2/aim1/BIDS_scans'
    args.parcellation_loc='./Parcels_Combo.nii.gz'
    args.output_dir = 'tmp/OAK/data/uh2/aim1/BIDS_scans/derivatives/1stlevel/beta_parcellations'


tasks = ['ANT', 'CCTHot', 'DPX', 'discountFix', 'motorSelectiveStop', 
                 'stopSignal', 'stroop', 'twoByTwo', 'WATT3']
bids_dir = args.bids_dir
output = args.output_dir

first_level_dir  = os.path.join(bids_dir,'derivatives', '1stlevel')   

parcellation_filename = args.parcellation_loc
masker = NiftiLabelsMasker(labels_img=parcellation_filename, 
standardize=True, memory='nilearn_cache', verbose=5)
subject_dirs = glob(os.path.join(first_level_dir, '*[!.html]')) # get subject dirs


for task in tasks:
    for subj in subject_dirs: #loop over paths
        #try:
            first_level = glob(os.path.join(subj, task,
            'maps_RT-True_beta-False', 'contrast-*.nii.gz'))
            subid = subj.split('/')[-1].split('-')[-1]
            print(first_level) 
            for cont in first_level: 
                beta_array = masker.fit_transform([cont])
                contrast = cont.split('/')[-1].replace('.nii.gz', '')

                pd.DataFrame(beta_array).to_csv(os.path.join(output,
                f'{subid}_{task}_{contrast}_.csv'))
        # except:
        #     print(f'{subj}, {task} failed')
import argparse
import sys
from nilearn import image
from glob import glob
import pandas as pd
import os
from nilearn.input_data import NiftiLabelsMasker

parser = argparse.ArgumentParser(description='First Level Entrypoint script')
parser.add_argument('-bids_dir', default='/data')
parser.add_argument('-parcellation_loc', default='./Parcels_Combo.nii.gz')
parser.add_argument('-output_dir', default='/data/derivatives/1stlevel/timeseries')
parser.add_argument('--tasks', nargs="+", help="Choose from ANT, CCTHot, discountFix, DPX, motorSelectiveStop, stopSignal, stroop, surveyMedley, twoByTwo, WATT3")

if '-bids_dir' in sys.argv or '-h' in sys.argv:
    args = parser.parse_args()
else:
    args = parser.parse_args([])
    args.bids_dir = 'tmp/OAK/data/uh2/aim1/BIDS_scans'
    args.parcellation_loc='./Parcels_Combo.nii.gz'
    args.output_dir = 'tmp/OAK/data/uh2/aim1/BIDS_scans/derivatives/1stlevel/timeseries'
    args.tasks = ['ANT', 'CCTHot', 'DPX', 'discountFix', 'motorSelectiveStop', 'stopSignal', 'stroop', 'twoByTwo', 'WATT3']


tasks = args.tasks
bids_dir = args.bids_dir
output = args.output_dir


fmriprep_label = 'fmriprep'

derivs_dir = os.path.join(bids_dir, 'derivatives')
fmriprep_dir = os.path.join(derivs_dir, fmriprep_label)

subject_dirs = glob(os.path.join(fmriprep_dir, '*[!.html]')) # get subject dirs


parcellation_filename = args.parcellation_loc
masker = NiftiLabelsMasker(labels_img=parcellation_filename, standardize=True, memory='nilearn_cache', verbose=5)

def preprocess_confounds(confound_df):
    # global signal
    del confound_df['global_signal']

    return confound_df.fillna(method='bfill') #remove nan

def get_timeseries(masker, img_path, confound_df):
    time_series = masker.fit_transform(img_path, confounds=confound_df.values)
    return time_series


for task in tasks:
    for subj in subject_dirs:
        try:
            bold_file = glob(os.path.join(subj, '*/func', f'*{task}*MNI*bold.nii.gz'))[0]
            confound_file = glob(os.path.join(subj, '*/func', f'*{task}*confounds_regressors.tsv'))[0]

            confound_df = preprocess_confounds(pd.read_csv(confound_file, delimiter='\t').copy())

            timeseries = get_timeseries(masker, bold_file, confound_df)

            subid = subj.split('/')[-1].split('-')[-1]
            pd.DataFrame(timeseries).to_csv(os.path.join(output, f'{subid}_{task}.csv'))
        except:
            print(f'file not found for task: {task} in {subj}')
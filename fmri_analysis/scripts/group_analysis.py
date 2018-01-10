import argparse
from glob import glob
import json
from multiprocessing import Pool 
from nipype.caching import Memory
from nipype.interfaces import fsl
from nilearn import datasets, image
from nilearn.regions import RegionExtractor
from nilearn.decomposition import CanICA
import numpy as np
import os
from os import makedirs
from os.path import join
import pandas as pd
from utils.utils import concat_and_smooth, get_contrast_names
from utils.utils import (create_neural_feature_mat, create_projections_df)
import re
import shutil

# parse arguments
parser = argparse.ArgumentParser(description='fMRI Analysis Entrypoint Script.')
parser.add_argument('output_dir', default = None, 
                    help='The directory where the output files '
                    'should be stored. These just consist of plots.')
parser.add_argument('--data_dir')
parser.add_argument('--mask_dir',)
parser.add_argument('--tasks',help='The tasks'
                   'that should be analyzed. Defaults to all.',
                   nargs="+")
args = parser.parse_args()

working_dir = join(args.output_dir, "workingdir")
output_dir = join(args.output_dir, "custom_modeling")

data_dir = '/Data'
if args.data_dir:
  data_dir = args.data_dir
  
mask_dir = None
if args.mask_dir:
    mask_dir = args.mask_dir

tasks = ['ANT', 'CCTHot', 'discountFix',
         'DPX', 'motorSelectiveStop',
         'stopSignal', 'stroop', 
         'surveyMedley', 'twoByTwo', 'WATT3']
if args.tasks:
    tasks = args.tasks

makedirs(output_dir, exist_ok=True)

# ********************************************************
# Create group maps
# ********************************************************
print('Creating Group Maps...')

# create mask over all tasks
# create 95% brain mask
mask_loc = join(output_dir, 'group_mask.nii.gz')
brainmasks = glob(join(mask_dir,'sub-s???',
                       '*','func',
                       '*MNI152NLin2009cAsym_brainmask*'))
mean_mask = image.mean_img(brainmasks)
group_mask = image.math_img("a>=0.8", a=mean_mask)
group_mask.to_filename(mask_loc)
    
def get_tmap(task):
    # set mask location
    mask_loc = join(output_dir, 'group_mask.nii.gz')
    # create task dir
    task_dir = join(output_dir,task)
    makedirs(task_dir, exist_ok=True)
    # get all contrasts
    contrast_path = glob(join(data_dir,'*%s/contrasts.pkl' % task))
    if len(contrast_path)>0:
        contrast_path = contrast_path[0]
    else:
        return # move to next iteration if no contrast files found
    contrast_names = get_contrast_names(contrast_path)
    
    print('Creating %s group map' % task)
    for i,contrast_name in enumerate(contrast_names):
        name = task + "_" + contrast_name
        # load, smooth, and concatenate contrasts
        map_files = sorted(glob(join(data_dir,
                                     '*%s/cope%s.nii.gz' % (task, i+1))))
        # save subject names in order on 1st iteration
        if i==0:
            subj_ids = [re.search('s[0-9][0-9][0-9]', file).group(0) 
                            for file in map_files]
            # see if these maps have been run before, and, if so, skip
            try:
                previous_ids = json.load(open(join(data_dir,
                                                   task,
                                                   'subj_ids.json'), 'r'))
                if previous_ids == subj_ids:
                    print('No new subjects added since last ran, skipping...')
                    break
                else:
                    json.dump(subj_ids, open(join(task_dir, 'subj_ids.json'),'w'))
            except FileNotFoundError:
                json.dump(subj_ids, open(join(task_dir, 'subj_ids.json'),'w'))
        # if there are map files, create group map
        if len(map_files) > 1:
            smooth_copes = concat_and_smooth(map_files, smoothness=4.4)

            copes_concat = image.concat_imgs(smooth_copes.values(),
                                             auto_resample=True)
            copes_loc = join(task_dir, "%s_copes.nii.gz" % name)
            copes_concat.to_filename(copes_loc)
                  
            # perform permutation test to assess significance
            mem = Memory(base_dir=working_dir)
            randomise = mem.cache(fsl.Randomise)
            randomise_results = randomise(in_file=copes_loc,
                                          mask=mask_loc,
                                          one_sample_group_mean=True,
                                          tfce=True, # look at paper
                                          vox_p_values=True,
                                          num_perm=5000)
            # save results
            tfile_loc = join(task_dir, "%s_raw_tfile.nii.gz" % name)
            tfile_corrected_loc = join(task_dir, "%s_corrected_tfile.nii.gz" 
                                       % name)
            raw_tfile = randomise_results.outputs.tstat_files[0]
            corrected_tfile = randomise_results.outputs.t_corrected_p_files[0]
            shutil.move(raw_tfile, tfile_loc)
            shutil.move(corrected_tfile, tfile_corrected_loc)
            
# create group maps
pool = Pool()
pool.map(get_tmap, tasks)
pool.close() 
pool.join()
    
# ********************************************************
# Set up parcellation
# ********************************************************

#******************* Estimate parcellation from data ***********************
print('Creating ICA based parcellation')

# get map files of interest (explicit contrasts)
map_files = []
for task in tasks: 
    contrast_path = sorted(glob(join(data_dir,'*%s/contrasts.pkl' % task)))
    if len(contrast_path)>0:
        contrast_path = contrast_path[0]
    else:
        continue # move to next iteration if no contrast files found
    contrast_names = get_contrast_names(contrast_path)
    for i, name in enumerate(contrast_names):
        # only get explicit contrasts (i.e. not vs. rest)
        if '-' in name or 'network' in name:
            map_files += sorted(glob(join(data_dir,
                                   '*%s/zstat%s.nii.gz' % (task, i+1))))

# group map files by subject
subject_ids = np.unique([f.split(os.sep)[-2].split('_')[0] for f in map_files])
subject_map_files = []
for s in subject_ids:
    subject_map_files.append(image.concat_imgs([f for f in map_files if s in f]))

    
n_components_list = [20,50,70]
for n_comps in n_components_list:
    ##  get components
    canica = CanICA(mask = group_mask, n_components=n_comps, 
                    smoothing_fwhm=4.4, memory=join(working_dir, "nilearn_cache"), 
                    memory_level=2, threshold=3., 
                    verbose=10, random_state=0) # multi-level components modeling across subjects
    
    canica.fit(map_files)
    masker = canica.masker_
    components_img = masker.inverse_transform(canica.components_)
    components_img.to_filename(join(output_dir, 
                                    'canica%s_explicit_contrasts.nii.gz' 
                                    % n_comps))
    

##************* Get parcellation from established atlas ************
## get smith parcellation
smith_networks = datasets.fetch_atlas_smith_2009()['rsn70']

# ********************************************************
# Helper functions
# ********************************************************



# ********************************************************
# Reduce dimensionality of contrasts
# ********************************************************
def split_index(projections_df):
    subj = [f.split('_')[0] for f in projections_df.index]
    contrast = ['_'.join(f.split('_')[1:]) for f in projections_df.index]
    projections_df.insert(0, 'subj', subj)
    projections_df.insert(1, 'contrast', contrast)
    
    
parcellation_files = [('smith70', smith_networks),
                      ('canica20', 
                       join(output_dir, 'canica20_explicit_contrasts.nii.gz')),
                      ('canica50', 
                       join(output_dir, 'canica50_explicit_contrasts.nii.gz')),
                       ('canica70', 
                       join(output_dir, 'canica70_explicit_contrasts.nii.gz'))
                       ]

for parcellation_name, parcellation_file in parcellation_files:
    projection_filey = join(output_dir, '%s_projection.json' % parcellation_name)
    mask_file = join(output_dir, 'group_mask.nii.gz')
    projections_df = create_projections_df(parcellation_file, mask_file, 
                                           data_dir, tasks, projection_filey)
    
    # create a subject x neural feature vector where each column is a component
    # for one contrast
    neural_feature_mat = create_neural_feature_mat(projections_df,
                                                   filename=join(output_dir, 
                                                        '%s_neural_features.json'  
                                                        % parcellation_name))

import argparse
from glob import glob
import json
from multiprocessing import Pool 
from nipype.caching import Memory
from nipype.interfaces import fsl
from nilearn import datasets, image
from nilearn.regions import RegionExtractor
import numpy as np
from os import makedirs
from os.path import join
import pandas as pd
from utils.utils import concat_and_smooth, get_contrast_names
from utils.utils import project_contrast
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
         'twoByTwo', 'WATT3']
if args.tasks:
    tasks = args.tasks

makedirs(output_dir, exist_ok=True)

# ********************************************************
# Create group maps
# ********************************************************
print('Creating Group Maps...')

print('Creating Group Mask...')
# create mask over all tasks
# create 95% brain mask
mask_loc = join(output_dir, 'group_mask.nii.gz')
brainmasks = glob(join(mask_dir,'sub-s???',
                       '*','func',
                       '*MNI152NLin2009cAsym_brainmask*'))
mean_mask = image.mean_img(brainmasks)
group_mask = image.math_img("a>=0.95", a=mean_mask)
group_mask.to_filename(mask_loc)
    
# define group mask function

def get_tmap(task, data_dir, output_dir, working_dir):
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
            except FileNotFoundError:
                json.dump(subj_ids, open(join(task_dir, 'subj_ids.json'),'w'))
        # if there are map files, create group map
        if len(map_files) > 1:
            smooth_copes = concat_and_smooth(map_files, smoothness=8)

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
                                          tfce=True,
                                          vox_p_values=True,
                                          num_perm=500)
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
from sklearn.decomposition import FastICA
from nilearn.decomposition import CanICA
from nilearn.input_data import NiftiMasker

# get map files of interest (explicit contrasts)
map_files = []
for task in tasks: 
    contrast_path = glob(join(data_dir,'*%s/contrasts.pkl' % task))
    if len(contrast_path)>0:
        contrast_path = contrast_path[0]
    else:
        continue # move to next iteration if no contrast files found
    contrast_names = get_contrast_names(contrast_path)
    for i, name in enumerate(contrast_names):
        # only get explicit contrasts (i.e. not vs. rest)
        if '-' in name:
            map_files += glob(join(data_dir,
                                   '*%s/zstat%s.nii.gz' % (task, i+1)))

n_components_list = [20,40]
for n_comps in n_components_list:
    ## using sklearn and nilearn
    #nifti_masker = NiftiMasker(mask_img=group_mask, smoothing_fwhm=4,
    #                           standardize=False)
    #compressor = FastICA(n_components=n_components, whiten=True)
    #masker_output = nifti_masker.fit_transform(map_files)
    #reduced = compressor.fit_transform(masker_output)
    
    
    # same thing with canICA
    canica = CanICA(mask = group_mask, n_components=n_comps, 
                    smoothing_fwhm=4., memory=join(working_dir, "nilearn_cache"), 
                    memory_level=2, threshold=3., 
                    verbose=10, random_state=0)
    
    canica.fit(map_files)
    masker = canica.masker_
    components_img = masker.inverse_transform(canica.components_)
    components_img.to_filename(join(output_dir, 
                                    'canica%s_explicit_contrasts.nii.gz' 
                                    % n_comps))


##************* Get parcellation from established atlas ************
## get smith parcellation
#smith_networks = datasets.fetch_atlas_smith_2009()['rsn20']
## create atlas
## ref: https://nilearn.github.io/auto_examples/04_manipulating_images/plot_extract_rois_smith_atlas.html
## this function takes whole brain networks and breaks them into contiguous
## regions. extractor.index_ labels each region as corresponding to one
## of the original brain maps
#
#extractor = RegionExtractor(smith_networks, min_region_size=800,
#                            threshold=98, thresholding_strategy='percentile')
#extractor.fit()
#regions_img = extractor.regions_img_


# ********************************************************
# Helper functions
# ********************************************************

def get_avg_corr(projection, subset1, subset2):
    subset_corr = projections_df.T.corr().filter(regex=subset1) \
                                       .filter(regex=subset2, axis=0)
    return subset_corr.mean().mean()


# ********************************************************
# Reduce dimensionality of contrasts
# ********************************************************
for n_comps in n_components_list:
    mask_file = join(output_dir, 'group_mask.nii.gz')
    parcellation_file = join(output_dir, 'canica%s_explicit_contrasts.nii.gz' % n_comps)
    # project contrasts into lower dimensional space    
    projections = []
    index = []
    for task in tasks:
        # get all contrasts
        contrast_path = glob(join(data_dir,'*%s/contrasts.pkl' % task))
        if len(contrast_path)>0:
            contrast_path = contrast_path[0]
        else:
            continue # move to next iteration if no contrast files found
        contrast_names = get_contrast_names(contrast_path)
        # for each contrast, project into space defined by parcellation file
        for i,name in enumerate(contrast_names):
            func_files = sorted(glob(join(data_dir, '*%s/zstat%s.nii.gz' 
                                          % (task, i+1))))
            TS, masker = project_contrast(func_files,
                                          parcellation_file, 
                                          mask_file)
            projections.append(TS)
            index += [re.search('s[0-9][0-9][0-9]',f).group(0)
                        + '_%s_%s' % (task, name)
                        for f in func_files]
    projections_df = pd.DataFrame(np.vstack(projections), index)
    projections_df.to_json(join(output_dir, 
                                'canica%s_projection.json' % n_comps))

# create matrix of average correlations across contrasts
contrasts = sorted(np.unique([i[5:] for i in projections_df.index]))
avg_corrs = np.zeros((len(contrasts), len(contrasts)))
for i, cont1 in enumerate(contrasts):
    for j, cont2 in enumerate(contrasts):
        avg_corrs[i,j] = get_avg_corr(projections_df, cont1, cont2)
avg_corrs = pd.DataFrame(avg_corrs, index=contrasts, columns=contrasts)

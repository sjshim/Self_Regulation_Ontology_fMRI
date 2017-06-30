import argparse
from glob import glob
from nipype.caching import Memory
from nipype.interfaces import fsl
from nilearn import datasets, image, input_data, plotting
from nilearn.regions import RegionExtractor
import numpy as np
from os import makedirs, rename
from os.path import join
import pandas as pd
import pickle
from utils.utils import concat_and_smooth, get_contrast_names
from utils.utils import project_contrast
from utils.display_utils import dendroheatmap_left
import re
import seaborn as sns

# parse arguments
parser = argparse.ArgumentParser(description='fMRI Analysis Entrypoint Script.')
parser.add_argument('output_dir', default = None, 
                    help='The directory where the output files '
                    'should be stored. These just consist of plots.')
parser.add_argument('--data_dir',help='The label(s) of the participant(s)'
                   'that should be analyzed. Multiple '
                   'participants can be specified with a space separated list.')
parser.add_argument('--mask_dir',help='The label(s) of the participant(s)'
                   'that should be analyzed. Multiple '
                   'participants can be specified with a space separated list.')
parser.add_argument('--tasks',help='The tasks'
                   'that should be analyzed. Defaults to all.')
args, unknown = parser.parse_known_args()

if args.output_dir:
    output_dir = join(args.output_dir, "custom_modeling")

data_dir = '/Data'
if args.data_dir:
  data_dir = args.data_dir
  
mask_dir = None
if args.mask_dir:
    mask_dir = args.mask_dir

tasks = ['ANT', 'discountFix',
         'DPX', 'motorSelectiveStop',
         'stopSignal', 'stroop', 
         'twoByTwo', 'WATT3']
if args.tasks:
    tasks = args.tasks
    
output_dir = '/home/ian/Experiments/expfactory/Self_Regulation_Ontology_fMRI/fmri_analysis/output/custom_modeling'
data_dir = '/mnt/Sherlock_Scratch/output_noRT/1stLevel'
mask_dir = '/mnt/Sherlock_Scratch/fmriprep/fmriprep/'

makedirs(output_dir, exist_ok=True)

# ********************************************************
# Create group maps
# ********************************************************
task = 'stroop'
# create 95% brain mask
brainmasks = glob(join(mask_dir,'sub-s???',
                       '*','func',
                       '*%s*MNI152NLin2009cAsym_brainmask*' % task))
mean_mask = image.mean_img(brainmasks)
group_mask = image.math_img("a>=0.95", a=mean_mask)
#plotting.plot_roi(group_mask)

# get all contrasts
contrast_path = glob(join(data_dir,'*%s/contrasts.pkl' % task))[0]
contrast_names = get_contrast_names(contrast_path)

for i,contrast_name in enumerate(contrast_names):
    name = task + "_" + contrast_name
    # load, smooth, and concatenate contrasts
    map_files = glob(join(data_dir,'*%s/cope%s.nii.gz' % (task, i+1)))
    smooth_copes = concat_and_smooth(map_files, smoothness=8)
    pickle.dump(smooth_copes, 
                open(join(output_dir, '%s_smooth_copes.pkl' % name), 'wb'))
    
    # create a 
    copes_concat = image.concat_imgs(smooth_copes.values(), auto_resample=True)
    copes_loc = join(output_dir, "%s_copes.nii.gz" % name)
    copes_concat.to_filename(copes_loc)
    
    # create group mask over relevant contrasts
    mask_loc = join(output_dir, "%s_group_mask.nii.gz" % task)
    group_mask = image.resample_to_img(group_mask, 
                                       copes_concat, interpolation='nearest')
    group_mask.to_filename(mask_loc)

    # perform permutation test to assess significance
    mem = Memory(base_dir='.')
    randomise = mem.cache(fsl.Randomise)
    randomise_results = randomise(in_file=copes_loc,
                                  mask=mask_loc,
                                  one_sample_group_mean=True,
                                  tfce=True,
                                  vox_p_values=True,
                                  num_perm=500)
    tfile_loc = join(output_dir, "%s_raw_tfile.nii.gz" % name)
    tfile_corrected_loc = join(output_dir, "%s_corrected_tfile.nii.gz" % name)
    raw_tfile = randomise_results.outputs.tstat_files[0]
    corrected_tfile = randomise_results.outputs.t_corrected_p_files[0]
    rename(raw_tfile, tfile_loc)
    rename(corrected_tfile, tfile_corrected_loc)
    
    # plotting.plot_stat_map('/home/ian/Experiments/expfactory/Self_Regulation_Ontology_fMRI/fmri_analysis/output/custom_modeling/incongruent-congruent_raw_tfile.nii.gz', threshold=2)
    
# ********************************************************
# Helper functions
# ********************************************************

# turn projections into dataframe
def projections_to_df(projections):
    all_projections = []
    index = []
    for k,v in projections.items():
    	all_projections.append(v)
    	index += [k]
    
    all_projections = np.vstack([i for i in all_projections])
    all_projections = pd.DataFrame(all_projections, index=index)
    sort_index = sorted(all_projections.index, key = lambda x: (x[4:8], 
                                                                x[-1],
                                                                x[0:4]))
    all_projections = all_projections.loc[sort_index]
    return all_projections


def get_avg_corr(projection, subset1, subset2):
    subset_corr = projections_df.T.corr().filter(regex=subset1) \
                                       .filter(regex=subset2, axis=0)
    return subset_corr.mean().mean()

# ********************************************************
# Set up parcellation
# ********************************************************

# get smith parcellation
smith_networks = datasets.fetch_atlas_smith_2009()['rsn20']
# create atlas
# ref: https://nilearn.github.io/auto_examples/04_manipulating_images/plot_extract_rois_smith_atlas.html
# this function takes whole brain networks and breaks them into contiguous
# regions. extractor.index_ labels each region as corresponding to one
# of the original brain maps

extractor = RegionExtractor(smith_networks, min_region_size=800,
                            threshold=98, thresholding_strategy='percentile')
extractor.fit()
regions_img = extractor.regions_img_

# ********************************************************
# Reduce dimensionality of contrasts
# ********************************************************

# project contrasts into lower dimensional space    
contrasts = range(12)
projections = {}
for task in tasks:
    for contrast in contrasts:
    	func_files = sorted(glob(join(data_dir, '*%s/zstat%s.nii.gz' \
                                   % (task, contrast))))
    	for func_file in func_files:
             subj = re.search('s[0-9][0-9][0-9]',func_file).group(0)
             TS, masker = project_contrast(func_file,smith_networks)
             projections[subj + '_' + task + '_zstat%s' % contrast] = TS
projections_df = projections_to_df(projections)
projections_df.to_json(join(output_dir, 'task_projection.json'))

# create matrix of average correlations across contrasts
contrasts = sorted(np.unique([i[-10:] for i in projections_df.index]))
avg_corrs = np.zeros((len(contrasts), len(contrasts)))
for i, cont1 in enumerate(contrasts):
    for j, cont2 in enumerate(contrasts):
        avg_corrs[i,j] = get_avg_corr(projections_df, cont1, cont2)
avg_corrs = pd.DataFrame(avg_corrs, index=contrasts, columns=contrasts)

# ********************************************************
# Plotting
# ********************************************************

# plot the inverse projection, sanity check
#plotting.plot_stat_map(masker.inverse_transform(projections['s192_stroop_cont4'])) 
    
## plots
#f, ax = sns.plt.subplots(1,1, figsize=(20,20))
#sns.heatmap(projections_df.T.corr(), ax=ax, square=True)
## dendrogram heatmap
#fig, leaves = dendroheatmap_left(projections_df.T.corr(), 
#                                 label_fontsize='small')
#fig.savefig(join(output_dir, 'task_dendoheatmap.png'))

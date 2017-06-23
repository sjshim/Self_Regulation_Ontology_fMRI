import argparse
from glob import glob
from nilearn import datasets, image, input_data, plotting
from nilearn.regions import RegionExtractor
import numpy as np
from os.path import join
import pandas as pd
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
args, unknown = parser.parse_known_args()

if args.output_dir:
    output_dir = args.output_dir

data_dir = '/Data'
if args.data_dir:
  data_dir = args.data_dir

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
# Helper functions
# ********************************************************

# function to get TS within labels
def project_contrast(contrast_file, parcellation_file):
	parcellation = image.load_img(parcellation_file)
	if len(parcellation.shape) == 3:
         masker = input_data.NiftiLabelsMasker(labels_img=parcellation_file, 
                                               resampling_target="labels", 
                                               standardize=False,
                                               memory='nilearn_cache', 
                                               memory_level=1)
	elif len(parcellation.shape) == 4:
         masker = input_data.NiftiMapsMasker(maps_img=parcellation_file, 
                                             resampling_target="maps", 
                                             standardize=False,
                                             memory='nilearn_cache',
                                             memory_level=1)
	time_series = masker.fit_transform(contrast_file)
	return time_series, masker

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
# Reduce dimensionality of contrasts
# ********************************************************

# project contrasts into lower dimensional space    
tasks = ['ANT', 'discountFix',
         'DPX', 'motorSelectiveStop',
         'stopSignal', 'stroop', 
         'twoByTwo', 'WATT3']
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
if output_dir:
    projections_df.to_json(join(output_dir, 'task_projection.json'))

# create matrix of average correlations across contrasts
contrasts = sorted(np.unique([i[-10:] for i in projections_df.index]))
avg_corrs = np.zeros((len(contrasts), len(contrasts)))
for i, cont1 in enumerate(contrasts):
    for j, cont2 in enumerate(contrasts):
        avg_corrs[i,j] = get_avg_corr(projections_df, cont1, cont2)
avg_corrs = pd.DataFrame(avg_corrs, index=contrasts, columns=contrasts)
print(avg_corrs.columns)

# ********************************************************
# Plotting
# ********************************************************

# plot the inverse projection, sanity check
#plotting.plot_stat_map(masker.inverse_transform(projections['s192_stroop_cont4'])) 
    
# plots
f, ax = sns.plt.subplots(1,1, figsize=(20,20))
sns.heatmap(projections_df.T.corr(), ax=ax, square=True)
# dendrogram heatmap
fig, leaves = dendroheatmap_left(projections_df.T.corr())
if output_dir:
    fig.save_fig(join(output_dir, 'task_dendoheatmap.png'))

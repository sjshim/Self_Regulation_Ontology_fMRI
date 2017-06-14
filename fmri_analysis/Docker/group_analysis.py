from glob import glob
from nilearn import datasets, image, input_data, plotting
from nilearn.regions import RegionExtractor
import numpy as np
import pandas as pd
from utils.display_utils import dendroheatmap_left
import re
import seaborn as sns

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

# extract time series for each functional run
tasks = ['ANT', 'DPX', 'motorSelectiveStop', 'stopSignal', 'stroop', 
         'twoByTwo', 'WATT3']
#func_files = glob('/mnt/temp/*/*/func/*%s*-MNI152NLin2009cAsym_preproc.nii.gz' % task)[0:4]
    
tasks=['stroop', 'stopSignal']
contrasts = [1,2,3,4]
projections = {}
for task in tasks:
    for contrast in contrasts:
    	func_files = sorted(glob('/mnt/Sherlock_Scratch/datasink/1stLevel/' + \
                           '*%s/cope%s.nii.gz' % (task, contrast)))
    	for func_file in func_files:
             subj = re.search('s[0-9][0-9][0-9]',func_file).group(0)
             TS, masker = project_contrast(func_file,smith_networks)
             projections[subj + '_' + task + '_cont%s' % contrast] = TS

# plot the inverse projection, sanity check
plotting.plot_stat_map(masker.inverse_transform(projections['s192_stroop_cont4']))

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

projections_df = projections_to_df(projections)

# plots
f, ax = sns.plt.subplots(1,1, figsize=(20,20))
sns.heatmap(projections_df.T.corr(), ax=ax, square=True)
# dendrogram heatmap
dendroheatmap_left(projections_df.T.corr())

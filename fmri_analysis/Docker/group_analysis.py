from glob import glob
from nilearn import image, input_data
from nilearn.regions import RegionExtractor
import numpy as np
import pandas as pd
import seaborn as sns

# get smith parcellation
from nilearn import datasets
smith = datasets.fetch_atlas_smith_2009()['rsn20']


# function to get TS within labels
def project_contrast(contrast_file, parcellation_file):
	parcellation = image.load_img(parcellation_file)
	if len(parcellation.shape) == 3:
		masker = input_data.NiftiLabelsMasker(labels_img=parcellation_file, 
			resampling_target="labels", standardize=False,
		    memory='nilearn_cache', memory_level=1)
	elif len(parcellation.shape) == 4:
		#extractor = RegionExtractor(parcellation_file, threshold=0.5,
        #                    thresholding_strategy='ratio_n_voxels',
        #                    extractor='local_regions',
        #                    standardize=True, min_region_size=1350)

		masker = input_data.NiftiMapsMasker(maps_img=parcellation_file, 
			resampling_target="maps", standardize=False,
		    memory='nilearn_cache', memory_level=1)
	time_series = masker.fit_transform(contrast_file)
	return time_series

# extract time series for each functional run
tasks = ['ANT', 'DPX', 'motorSelectiveStop', 'stopSignal', 'stroop', 'twoByTwo', 'WATT3']
#func_files = glob('/mnt/temp/*/*/func/*%s*-MNI152NLin2009cAsym_preproc.nii.gz' % task)[0:4]

task='stroop'
projections = {}
for contrast in [1,2,3,4]:
	func_files = glob('/mnt/Sherlock_Scratch/datasink/1stLevel/*%s/tstat%s.nii.gz' % (task, contrast))
	group_TS = []
	for func_file in func_files:
		time_series = project_contrast(func_file,smith)
		group_TS.append(time_series)
	contrast_projection = np.vstack(group_TS)
	projections[task + '_cont%s' % contrast] = contrast_projection

all_projections = []
index = []
for k,v in projections.items():
	n_subjs = v.shape[0]
	all_projections.append(v)
	index += [k]*n_subjs

all_projections = np.vstack([i for i in all_projections])
all_projections = pd.DataFrame(all_projections, index=index)


# coding: utf-8

# In[ ]:


import argparse
from functools import partial
from glob import glob
from itertools import combinations
from joblib import Parallel, delayed
import json
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import nibabel
import nilearn
from nilearn import datasets, image
import numpy as np
from os import makedirs, path, sep
import pandas as pd
import pickle
import shutil
import sys

from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
from scipy.spatial.distance import squareform

from utils.secondlevel_utils import *
from utils.secondlevel_plot_utils import *


# In[ ]:


# imports for plotting exploring
import matplotlib
matplotlib.use("agg")
import seaborn as sns
import matplotlib.pyplot as plt
from nilearn import plotting
import warnings
warnings.simplefilter("ignore", UserWarning)


# # Parse Arguments
# These are not needed for the jupyter notebook, but are used after conversion to a script for production
# 
# - conversion command:
#   - jupyter nbconvert --to script --execute 2ndlevel_analysis.ipynb

# In[ ]:


parser = argparse.ArgumentParser(description='fMRI Analysis Entrypoint Script.')
parser.add_argument('-derivatives_dir')
parser.add_argument('-working_dir', default=None)
parser.add_argument('--tasks', nargs="+")
parser.add_argument('--n_procs', default=4, type=int)
parser.add_argument('--num_perm', default=1000, type=int, help="Passed to fsl.randomize")
parser.add_argument('--ignore_rt', action='store_false')
parser.add_argument('--rerun', action='store_true')
parser.add_argument('--mask_threshold', default=.8, type=float)
if '-derivatives_dir' in sys.argv or '-h' in sys.argv:
    matplotlib.use("agg")
    args = parser.parse_args()
else:
    # if run as a notebook reduce the set of args
    args = parser.parse_args([])
    args.derivatives_dir='/data/derivatives'
    args.rerun = False


# In[ ]:


derivatives_dir = args.derivatives_dir
fmriprep_dir = path.join(derivatives_dir, 'fmriprep', 'fmriprep')
first_level_dir = path.join(derivatives_dir, '1stlevel')
second_level_dir = path.join(derivatives_dir,'2ndlevel')
if args.working_dir is None:
    working_dir = path.join(derivatives_dir, '2ndlevel_workingdir')
else:
    working_dir = path.join(args.working_dir, '2ndlevel_workingdir')
makedirs(working_dir, exist_ok=True)
    
tasks = ['ANT', 'CCTHot', 'discountFix',
         'DPX', 'motorSelectiveStop',
         'stopSignal', 'stroop', 
         'surveyMedley', 'twoByTwo', 'WATT3']
if args.tasks:
    tasks = args.tasks
regress_rt = args.ignore_rt
model = 'model-rt' if regress_rt == True else 'model-nort'
mask_threshold = args.mask_threshold


# # Create Group Mask

# In[ ]:


# create mask over all tasks
# create 95% brain mask
mask_loc = path.join(second_level_dir, 'group_mask_thresh-%s.nii.gz' % str(mask_threshold))
if path.exists(mask_loc) == False or args.rerun:
    create_group_mask(mask_loc, fmriprep_dir, mask_threshold)


# In[ ]:


# % matplotlib inline
plotting.plot_img(mask_loc, title='Group Mask, Threshold: %s%%' % str(mask_threshold*100))


# # Setup

# In[ ]:


# set up functions with some default parameters
get_group_maps = partial(get_group_maps, second_level_dir=second_level_dir,
                        tasks=tasks, model=model)
get_ICA_parcellation = partial(get_ICA_parcellation, second_level_dir=second_level_dir,
                               mask_loc=mask_loc, working_dir=working_dir)


# In[ ]:


# get data files
file_type = 'zstat'
map_files = get_map_files(map_prefix=file_type, 
                          first_level_dir=first_level_dir,
                        tasks=tasks, model=model, selectors=None)
contrast_names = list(map_files.keys())
# reduce the number of files to make execution quicker for testing
def random_sample(lst, n):
    return [lst[i] for i in np.random.choice(range(len(lst)), n, replace=False)]
metadata = get_metadata(map_files)


# In[ ]:


# remove empty contrasts
items = list(map_files.items())
for key, val in items:
    if len(val) == 0:
        del map_files[key]
        print('No contrasts found for %s!' % key)
contrast_names = list(map_files.keys())


# In[ ]:


"""
#iterative version
concat_out = concat_map_files(map_files, file_type=individual_file_type,
                                second_level_dir=second_level_dir, model=model,
                                verbose=True)
"""
# concat files in parallel
concat_map_files = partial(concat_map_files, file_type=file_type,
                           second_level_dir=second_level_dir, model=model, verbose=False,
                          rerun=args.rerun)

list_dicts = [{k:map_files[k]} for k in map_files.keys()]
concat_out = Parallel(n_jobs=args.n_procs)(delayed(concat_map_files)(task) for task in list_dicts)
concat_out = flatten(concat_out)


# In[ ]:


# get the average  map for each contrast
to_extract = concat_out
group_map_files = get_mean_maps(to_extract, contrast_names, save=True, rerun=args.rerun)
group_meta = get_metadata(group_map_files)


# # Create Group Maps

# In[ ]:


# #iterative version
# smooth_out = smooth_concat_files(concated_map_files, verbose=True)
# smooth files in parallel
smooth_concat_files = partial(smooth_concat_files, verbose=False, fwhm=6.6, rerun=args.rerun)
smooth_out = Parallel(n_jobs=args.n_procs)(delayed(smooth_concat_files)([concat_file]) for concat_file in concat_out)
smooth_out = flatten(smooth_out)


# In[ ]:


# then tmap
contrast_tmap_parallel = partial(save_tmaps, mask_loc=mask_loc, working_dir=working_dir, 
                                 permutations=args.num_perm, rerun=args.rerun)
tmap_out = Parallel(n_jobs=args.n_procs)(delayed(contrast_tmap_parallel)(filey) for filey in smooth_out)
tmap_raw, tmap_correct = zip(*tmap_out)


# In[ ]:


task_contrast_dirs = sorted(glob(path.join(second_level_dir, '*', 'model-rt', 'wf-contrast')))
for d in task_contrast_dirs:
    plot_2ndlevel_maps(d, lookup='*raw*', threshold=.95) # *raw* for raw


# # Searchlight RSA

# In[ ]:


print('Beginning Searchlight Analysis')
searchlight_dir = path.join(second_level_dir, 'Extracted_Data', 'searchlight')
makedirs(path.join(searchlight_dir, 'Plots'), exist_ok=True)


# In[ ]:


from nilearn.input_data.nifti_spheres_masker import _apply_mask_and_get_affinity
from nilearn.input_data import NiftiSpheresMasker
from scipy.spatial.distance import pdist

def get_voxel_coords(mask_loc):
    mask, mask_affine = masking._load_mask_img(mask_loc)
    mask_coords = np.where(mask != 0)
    process_mask_coords = image.resampling.coord_transform(
            mask_coords[0], mask_coords[1],
            mask_coords[2], mask_affine)
    process_mask_coords = np.asarray(process_mask_coords).T
    return process_mask_coords, mask

def RSA(list_rows, X):
    subset = X[:,list_rows]
    return pdist(X[:,list_rows], metric='correlation')

def searchlight_RSA(imgs, mask_loc, radius=10, n_jobs=4):
    voxel_coords, mask = get_voxel_coords(mask_loc)
    X, A = _apply_mask_and_get_affinity(voxel_coords, 
                                        imgs, 
                                        radius=radius, allow_overlap=True,
                                        mask_img=mask_loc)
    RDMs = Parallel(n_jobs=n_jobs)(
                    delayed(RSA)(A.rows[list_i], X) for list_i in range(voxel_coords.shape[0]))
    return RDMs, mask


# In[ ]:


group_searchlight_file = path.join(searchlight_dir, 'groupcontrasts_searchlight_RDM.pkl')
imgs = image.concat_imgs(group_map_files.values())
if path.exists(group_searchlight_file) and not args.rerun:
    RDMs, mask = pickle.load(open(group_searchlight_file, 'rb'))
else:
    RDMs, mask = searchlight_RSA(imgs, mask_loc)
    pickle.dump([RDMs, mask], open(group_searchlight_file, 'wb'))


# In[ ]:


pca = PCA(3)
vectorized_RDMs = np.vstack(RDMs)
pca_RDMs = pca.fit_transform(vectorized_RDMs)
scaled = minmax_scale(pca_RDMs)


# In[ ]:


for i in range(pca.n_components):
    RDM_3d = np.zeros(mask.shape)
    values = np.zeros(np.sum(mask))
    values[:len(RDMs)] = scaled[:,i]
    RDM_3d[mask] = values
    RDM_3d = image.new_img_like(mask_loc, RDM_3d)
    # html for surface
    view = plotting.view_img_on_surf(RDM_3d)
    view.save_as_html(path.join(searchlight_dir, 'Plots', 'RSA_PCA%s_surface.html' % str(i+1)))
    # pdf for volume
    f = plt.figure(figsize=(30,10))
    plotting.plot_stat_map(RDM_3d, figure=f, title='RSA PCA %s' % str(i+1))
    f.savefig(path.join(searchlight_dir, 'Plots', 'RSA-PCA%s_volume.pdf' % str(i+1)))


# In[ ]:


# we can also visualize the RDMs reflecting each of these first 3 components
n_cols = 3
n_rows = pca.n_components//n_cols
index = group_meta.apply(lambda x: '_'.join(x), axis=1)
for i, component in enumerate(pca.components_):
    component_RDM = squareform(component)
    component_RDM = pd.DataFrame(component_RDM, index=index, columns=index)
    f = sns.clustermap(component_RDM)
    f.savefig(path.join(searchlight_dir, 'Plots', 'RSA-PCA%s_clustermap.pdf' % str(i+1)))


# # Parcellations, Atlases and RDM
# 
# Projecting into a lower dimensional space allows the evaluation of whole-brain similarity analysis (clustering)
# 
# RDMs can also be evaluated within parcellation regions

# ## Get parcellations to use

# In[ ]:


print('Getting Parcellation')
parcellation_dir = path.join(second_level_dir, 'parcellation')
makedirs(parcellation_dir, exist_ok=True)
"""
# calculate ICA parcel
n_comps = 20; ICA_prefix = 'contrast'
ICA_path = path.join(parcellation_dir, '%s_canica%s.nii.gz' % (ICA_prefix, n_comps))
if path.exists(ICA_path) and not args.rerun:
    ICA_parcel = image.load_img(path.join(parcellation_dir, '%s_canica%s.nii.gz' % (ICA_prefix, n_comps)))
else:
    ICA_parcel = get_ICA_parcellation(map_files, n_comps=n_comps, file_name=ICA_prefix)
"""
# get literature parcels
target_img = list(map_files.values())[0] # get image to resample atlases to
harvard = get_established_parcellation("Harvard_Oxford", target_img=target_img, parcellation_dir=parcellation_dir)
#smith = get_established_parcellation("smith", target_img=target_img, parcellation_dir=parcellation_dir)
#glasser = get_established_parcellation("glasser", target_img=target_img, parcellation_dir=parcellation_dir)


# In[ ]:


# %matplotlib inline
# plotting.plot_prob_atlas(harvard_parcel)
# # what is RegionExtractor?


# ## Use parcellation to create ROIs and calculate RDMs amongst contrasts within each ROI

# Set up hyper parameters

# In[ ]:


parcel, parcel_labels, parcel_name, threshold = harvard
roi_extraction_dir = second_level_dir
extraction_dir = path.join(second_level_dir, 'Extracted_Data', 'parcel-%s' % parcel_name)
makedirs(path.join(extraction_dir, 'Plots'), exist_ok=True)
print('Using Parcellation: %s' % parcel_name)


# In[ ]:


if len(parcel.shape) == 4:
    plotting.plot_prob_atlas(parcel, title="Parcellation", cut_coords=[0, -41, 10])
else:
    plotting.plot_roi(parcel, title="Parcellation", cut_coords=[0, -41, 10])


# ### Calculate RDMs for each region for each group map

# In[ ]:


print('Calculating RDMs for each RDM based on group contrasts')
group_extraction_file = path.join(extraction_dir, 'groupcontrasts_extraction.pkl')
if path.exists(group_extraction_file) and not args.rerun:
    group_roi_contrasts = pickle.load(open(group_extraction_file, 'rb'))
else:
    group_roi_contrasts = extract_roi_vals(group_map_files, parcel, extraction_dir, 
                                   threshold=threshold, metadata=group_meta, 
                                   labels=parcel_labels, rerun=args.rerun, 
                                   n_procs=args.n_procs, save=False)
    tmp = odict()
    for k,v in zip(parcel_labels, group_roi_contrasts):
        tmp[k] = v
    group_roi_contrasts = tmp
    pickle.dump(group_roi_contrasts, open(group_extraction_file, 'wb'))


# In[ ]:


group_RDMs = get_RDMs(group_roi_contrasts)
keys = [k for k,v in group_RDMs.items() if v is not None]


# In[ ]:


# plot random RDM
label = np.random.choice(keys)
index = parcel_labels.index(label)
roi = get_ROI_from_parcel(parcel, index, threshold)
RDM = group_RDMs[label]
if RDM is not None:
    RDM = pd.DataFrame(RDM, index=group_map_files.keys())
    plot_RDM(RDM, roi, title=label, cluster=True)


# #### RDM of RDMs
# 
# Each ROI has an RDM reflecting its "representation" of cognitive faculties probed by these contrasts. We can look at the similarity of RDMs to get a sense of the similarity of the cognitive fingerprint of individual regions

# In[ ]:


print('Group RDM of RDMs')
def tril(square_mat):
    return square_mat[np.tril_indices_from(square_mat, -1)]

# similarity of RDMs
vectorized_RDMs = np.vstack([tril(group_RDMs[k]) for k in keys if group_RDMs[k] is not None])
vectorized_RDMs = pd.DataFrame(vectorized_RDMs, index=keys)
RDM_of_RDMs = 1-vectorized_RDMs.T.corr()


# In[ ]:


# visualize RDM of RDMs
clustermap = sns.clustermap(RDM_of_RDMs, figsize=[15,15])
clustermap.savefig(path.join(extraction_dir, 'Plots', 'groupcontrasts_RDM_of_RDMs.pdf'))


# #### PCA of RDMs

# In[ ]:


print('PCA Colored Map of RDMs')
pca = PCA(3)
pca_RDMs = pd.DataFrame(pca.fit_transform(vectorized_RDMs), index=vectorized_RDMs.index)
scaled = minmax_scale(pca_RDMs)

# find indices of skipped ROIs (for whatever reason) and set them to 0
for i,label in enumerate(parcel_labels):
    if label not in keys:
        scaled = np.insert(scaled, i, [0]*pca.n_components, axis=0)
        
# we can then color the first 3 PCA components (RGB) and create color mixtures reflecting the RDM signature
colors = np.array([[1,0,0], [0,1,0], [0,0,1]])
def combined_colors(array, colors=colors):
    return np.dot(colors, array)
colored_pca = np.apply_along_axis(combined_colors, 1, scaled)

# create colormaps
all_color_list = colored_pca
PC1_color_list = [[0,0,0]]+[colors[0]*i for i in scaled[:,0]]
PC2_color_list = [[0,0,0]]+[colors[1]*i for i in scaled[:,1]]
PC3_color_list = [[0,0,0]]+[colors[2]*i for i in scaled[:,2]]


# In[ ]:


# compare RDM after dimensional reduction
i = np.random.randint(vectorized_RDMs.shape[0])
f, axes = plt.subplots(1,2,figsize=(12,5))
orig = vectorized_RDMs.iloc[i,:]
reconstruction = pca.inverse_transform(pca_RDMs.iloc[i,:])
corr = np.corrcoef(orig, reconstruction)[0,1]
sns.heatmap(squareform(orig), ax=axes[0])
sns.heatmap(squareform(reconstruction), ax=axes[1])
axes[0].set_title(vectorized_RDMs.index[i])
axes[1].set_title('PCA (%s) reconstruction, Corr: %0.2f' % (str(pca.n_components), corr))


# In[ ]:


# we can also visualize the RDMs reflecting each of these first 3 components
n_cols = 3
n_rows = pca.n_components//n_cols
index = group_meta.apply(lambda x: '_'.join(x), axis=1)
for i, component in enumerate(pca.components_):
    component_RDM = squareform(component)
    component_RDM = pd.DataFrame(component_RDM, index=index, columns=index)
    f = sns.clustermap(component_RDM)
    f.savefig(path.join(extraction_dir, 'Plots', 'groupcontrasts_RSA-PCA%s_clustermap.pdf' % str(i+1)))


# ##### Visualize in the Volume

# In[ ]:


cm = ListedColormap(all_color_list)
f=plt.figure(figsize=(12,8))
if len(parcel.shape) == 4:
    plotting.plot_prob_atlas(parcel, cmap=cm, view_type='filled_contours', figure=f, 
                             title="RDM -> PCA -> colors (1: Red, 2: Green, 3: Blue)")
else:
    plotting.plot_roi(parcel, cmap=cm, figure=f, 
                      title="RDM -> PCA -> colors (1: Red, 2: Green, 3: Blue)")
    f.savefig(path.join(extraction_dir, 'Plots', 'groupcontrasts_RSA-colored_volume.pdf' % str(i+1)))


# ##### Visualize on the Surface

# In[ ]:


# # plot on surface https://nilearn.github.io/plotting/index.html
# fsaverage = datasets.fetch_surf_fsaverage(data_dir=parcellation_dir, mesh='fsaverage')
# surf_mesh_l = fsaverage['infl_left']
# surf_mesh_r = fsaverage['infl_right']
# surf_projection = '/data/derivatives/2ndlevel/parcellation/glasser/lh.HCP-MMP1.annot'
# surf_projection = nibabel.freesurfer.read_annot(surf_projection, orig_ids=True)[0]


# In[ ]:


# def convert_to_stat_map(parcel, values):
#     data = parcel.get_data().copy()
#     for i in range(1, np.max(data)+1):
#         data[data==i] = values[i-1]
#     return image.new_img_like(parcel, data)


# In[ ]:


# # plot stat map on surface
# n=pca.n_components
# f, axes = plt.subplots(n,4, figsize=(24,6*n), subplot_kw={'projection': '3d'})
# f.subplots_adjust(hspace=-.25)
# f.subplots_adjust(wspace=-.2)
# cmaps = ['Reds', 'Greens', 'Blues']
# for i in range(n):
#     cm = 'cold_hot'
#     values = [row[i]*100 for row in scaled]
#     to_plot = convert_to_stat_map(parcel,values)
#     # left
#     texture = nilearn.surface.vol_to_surf(to_plot, fsaverage.pial_left)
#     _ = plotting.plot_surf_stat_map(surf_mesh_l, texture, hemi='left', axes=axes[i,0], figure=f, cmap=cm)
#     _ = plotting.plot_surf_stat_map(surf_mesh_l, texture, hemi='left', view='medial', axes=axes[i,1], figure=f, cmap=cm)
#     # right
#     texture = nilearn.surface.vol_to_surf(to_plot, fsaverage.pial_right)
#     _ = plotting.plot_surf_stat_map(surf_mesh_r, texture, hemi='right', view='medial', axes=axes[i,2], figure=f, cmap=cm)
#     _ = plotting.plot_surf_stat_map(surf_mesh_r, texture, hemi='right', axes=axes[i,3], figure=f, cmap=cm)
# f.savefig(path.join(extraction_dir, 'Plots', 'groupcontrasts_PC1-3.pdf'))


# In[ ]:


# # plot glasser annotation
# f1=plt.figure(figsize=(12,8))
# glassar_surface = plotting.plot_surf_stat_map(surf_mesh_l, surf_projection, figure=f1)


# ### Calculate RDMs for each region for each subject-contrast

# In[ ]:


# files = extract_roi_vals(to_extract, parcel, extraction_dir, 
#                  threshold=threshold, metadata=metadata, 
#                  labels=parcel_labels, rerun=args.rerun, 
#                  n_procs=1)


# #### Apply parcellations

# In[ ]:


"""
# ********************************************************
# Set up parcellation
# ********************************************************

#******************* Estimate parcellation from data ***********************
print('Creating ICA based parcellation')


# group map files by subject
subject_ids = np.unique([f.split(os.sep)[-2].split('_')[0] for f in map_files])
subject_map_files = []
for s in subject_ids:
    subject_map_files.append(image.concat_imgs([f for f in map_files if s in f]))






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
"""


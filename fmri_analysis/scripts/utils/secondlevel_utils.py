from collections import OrderedDict as odict
import glob as glob
import numpy as np
from os import path
import pandas as pd
import re

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

#fmri imports
import nibabel
from nilearn import datasets, image, input_data
from nilearn.decomposition import CanICA

# ********************************************************
# Functions to get fmri maps and get/create parcellations
# ********************************************************
def get_map_files(first_level_dir,
                  tasks,
                  model,
                  map_prefix='zstat',
                  selectors=None):
    map_files = odict()
    for task in tasks: 
        subjectinfo_paths = sorted(glob(path.join(first_level_dir,'*', task, model, 'wf-contrast', 'subjectinfo.pkl')))
        if len(subjectinfo_paths)>0:
            contrast_names = get_contrast_names(subjectinfo_paths[0])
        else:
            continue # move to next iteration if no contrast files found
        if contrast_names is None:
            continue

        # select only a subset of contrasts (i.e.  get explicit contrasts, not vs rest)
        if selectors is None:
            selectors = ['-', 'network', 'response_time']
        for i, name in enumerate(contrast_names):
            if np.logical_or.reduce([sel in name for sel in selectors]):
                map_files[task+'_'+name] = sorted(glob(path.join(first_level_dir,
                                                                 '*', 
                                                                 task,
                                                                 model,
                                                                 'wf-contrast', 
                                                                 'zstat%s.nii.gz' % str(i+1))))
    return map_files

def flatten_map_files(map_files):
    return [item for sublist in map_files.values() for item in sublist]

def get_ICA_parcellation(map_files,
                         mask_loc,
                         working_dir,
                         second_level_dir,
                         n_comps=20,
                         smoothing=4.4,
                         file_name=''):
    if type(map_files) == dict:
        map_files = flatten_map_files(map_files)
    group_mask = nibabel.load(mask_loc)
    ##  get components
    canica = CanICA(mask = group_mask, n_components=n_comps, 
                    smoothing_fwhm=smoothing, memory=path.join(working_dir, "nilearn_cache"), 
                    memory_level=2, threshold=3., 
                    verbose=10, random_state=0) # multi-level components modeling across subjects
    canica.fit(map_files)
    masker = canica.masker_
    components_img = masker.inverse_transform(canica.components_)
    components_img.to_filename(path.join(second_level_dir, 
                                        '%s_canica%s.nii.gz' 
                                        % (file_name, n_comps)))
    return components_img
    
def get_established_parcellation(parcellation="Harvard_Oxford", target_img=None):
    if parcellation == "Harvard_Oxford":
        data = datasets.fetch_atlas_harvard_oxford('cort-prob-2mm')
        parcel = nibabel.load(data['maps'])
        labels = data['labels'][1:] # first label is background
    if parcellation == "smith":
        data = datasets.fetch_atlas_smith_2009()['rsn70']
        parcel = nibabel.load(data)
        labels = range(parcel.shape[-1])
    if target_img:
        parcel = image.resample_to_img(parcel, target_img)
    return parcel, labels

# ********************************************************
# Functions to extract ROIs from parcellations
# ********************************************************
def get_ROI_from_parcel(parcel, ROI, threshold):
    # convert a probabilistic parcellation into an ROI mask
    roi_mask = parcel.get_data()[:,:,:,ROI]>threshold 
    roi_mask = image.new_img_like(parcel, roi_mask)
    return roi_mask

def extract_roi_vals(map_files, parcel, threshold, labels=None):
    """ Mask nifti images using a parcellation"""
    if type(map_files) == dict:
        map_files = flatten_map_files(map_files)
    roi_vals = odict()
    for roi_i in range(parcel.shape[-1]):
        roi_masker = input_data.NiftiMasker(get_ROI_from_parcel(parcel, roi_i, threshold))
        if labels:
            key = labels[roi_i]
        else:
            key = roi_i
        roi_vals[key] = roi_masker.fit_transform(map_files)
    return roi_vals

# ********************************************************
# RDM functions
# ********************************************************
def get_RDMs(ROI_dict):
    # converts ROI dictionary (returned by extract_roi_vals) of contrast X voxel values to RDMs
    RDMs = {}
    for key,val in ROI_dict.items():
        RDMs[key] = 1-np.corrcoef(val)
    return RDMs
        
# ********************************************************
# 2nd level analysis utility functions
# ********************************************************

def concat_and_smooth(map_files, smoothness=None):
    """
    Loads and smooths files specified in 
    map_files and creates a dictionary of them
    """
    smooth_copes = odict()
    for img_i, img in enumerate(sorted(map_files)):
        subj = re.search('s[0-9][0-9][0-9]',img).group(0)
        smooth_cope = image.smooth_img(img, smoothness)
        smooth_copes[subj] = smooth_cope
    return smooth_copes

# function to get TS within labels
def project_contrast(img_files, parcellation, mask_file):
    if type(parcellation) == str:
        parcellation = image.load_img(parcellation)
    resampled_images = image.resample_img(img_files, parcellation.affine)
    if len(parcellation.shape) == 3:
        masker = input_data.NiftiLabelsMasker(labels_img=parcellation, 
                                               resampling_target="labels", 
                                               standardize=False,
                                               memory='nilearn_cache', 
                                               memory_level=1)
    elif len(parcellation.shape) == 4:
         masker = input_data.NiftiMapsMasker(maps_img=parcellation, 
                                             mask_img=mask_file,
                                             resampling_target="maps", 
                                             standardize=False,
                                             memory='nilearn_cache',
                                             memory_level=1)
    time_series = masker.fit_transform(resampled_images)
    return time_series, masker

def create_projections_df(parcellation, mask_file, 
                         data_dir, tasks, filename=None):
    
    # project contrasts into lower dimensional space    
    projections = []
    index = []
    for task in tasks:
        task_files = get_map_files(tasks=[task])
        # for each contrast, project into space defined by parcellation file
        for contrast_name, func_files in task_files.items():
            TS, masker = project_contrast(func_files,
                                          parcellation, 
                                          mask_file)
            projections.append(TS)
            index += [re.search('s[0-9][0-9][0-9]',f).group(0)
                        + '_%s' % (contrast_name)
                        for f in func_files]
    projections_df = pd.DataFrame(np.vstack(projections), index)
    
    # split index into column names
    subj = [i[:4] for i in projections_df.index]
    contrast = [i[5:] for i in projections_df.index]
    projections_df.insert(0, 'subj', subj)
    projections_df.insert(0, 'contrast', contrast)
    
    # save
    if filename:
        projections_df.to_json(filename)
    return projections_df

# functions on projections df
def create_neural_feature_mat(projections_df, filename=None):
    # if projections_df is a string, load the file
    if type(projections_df) == str:
        assert path.exists(projections_df)
        projections_df = pd.read_json(projections_df)
        
    subj = [i[:4] for i in projections_df.index]
    contrast = [i[5:] for i in projections_df.index]
    projections_df.insert(0, 'subj', subj)
    projections_df.insert(0, 'contrast', contrast)
    neural_feature_mat = projections_df.pivot(index='subj', columns='contrast')
    if filename:
        neural_feature_mat.to_json(filename)
    return neural_feature_mat

def projections_corr(projections_df, remove_global=True, grouping=None):
    """ Create a correlation matrix of a projections dataframe
    
    Args:
        projections_df: a projection_df, as create by create_projection_df
        remove_global: if True, subtract the mean contrast
        grouping: "subj" or "contrast". If provided, average over the group
        
    Returns:
        Correlation Matrix
    """
    # if projections_df is a string, load the file
    if type(projections_df) == str:
        assert path.exists(projections_df)
        projections_df = pd.read_json(projections_df)
    
    if remove_global:
        projections_df.iloc[:,2:] -= projections_df.mean()
    if grouping:
        projections_df = projections_df.groupby(grouping).mean()
    return projections_df.T.corr()

def get_confusion_matrix(projections_df, normalize=True):
    # if projections_df is a string, load the file
    if type(projections_df) == str:
        assert path.exists(projections_df)
        projections_df = pd.read_json(projections_df)
        
    X = projections_df.iloc[:, 2:]
    y = projections_df.contrast
    clf = LogisticRegressionCV(multi_class='multinomial')
    predict = cross_val_predict(clf, X, y, cv=10)
    cm = confusion_matrix(y, predict)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm
    
                                           

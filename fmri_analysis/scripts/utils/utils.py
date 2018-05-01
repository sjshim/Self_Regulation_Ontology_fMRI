"""
some util functions
"""
from collections import OrderedDict as odict, defaultdict
from glob import glob
import nilearn
from nilearn import image, input_data
import numpy as np
from os.path import join, exists, sep
import pandas as pd
import pickle
import re
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

# ********************************************************
# Basic Help Methods
# ********************************************************

def get_event_dfs(data_dir, subj='', task=''):
    event_paths = glob(join(data_dir, '*%s*' % subj,
                                '*', 'func', '*%s*event*' % task))
    event_dfs = defaultdict(dict)
    for event_path in event_paths:
        subj = event_path.split(sep)[-4].replace('sub-','')
        task = event_path.split(sep)[-1].split('_')[-3][5:]
        event_df = pd.read_csv(event_path, sep='\t')
        event_dfs[subj][task] = event_df
    return event_dfs
    
def load_atlas(atlas_path, atlas_label_path=None):
    out = {}
    out['maps'] = atlas_path
    if atlas_label_path:
        file_data = np.loadtxt(atlas_label_path, 
                               dtype={'names': ('index', 'label'),
                                      'formats': ('i4', 'S50')})
        out['labels'] = [i[1].decode('UTF-8') for i in file_data]
    return out

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
        smooth_cope = nilearn.image.smooth_img(img, smoothness)
        smooth_copes[subj] = smooth_cope
    return smooth_copes

def get_contrast_names(contrast_path):
    contrasts = pickle.load(open(contrast_path,'rb'))
    contrast_names = [c[0] for c in contrasts]
    return contrast_names

# function to get TS within labels
def project_contrast(img_files, parcellation_file, mask_file):
    parcellation = image.load_img(parcellation_file)
    resampled_images = image.resample_img(img_files, parcellation.affine)
    if len(parcellation.shape) == 3:
        masker = input_data.NiftiLabelsMasker(labels_img=parcellation_file, 
                                               resampling_target="labels", 
                                               standardize=False,
                                               memory='nilearn_cache', 
                                               memory_level=1)
    elif len(parcellation.shape) == 4:
         masker = input_data.NiftiMapsMasker(maps_img=parcellation_file, 
                                             mask_img=mask_file,
                                             resampling_target="maps", 
                                             standardize=False,
                                             memory='nilearn_cache',
                                             memory_level=1)
    time_series = masker.fit_transform(resampled_images)
    return time_series, masker

def create_projections_df(parcellation_file, mask_file, 
                         data_dir, tasks, filename=None):
    
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
        assert exists(projections_df)
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
        assert exists(projections_df)
        projections_df = pd.read_json(projections_df)
    
    if remove_global:
        projections_df.iloc[:,2:] -= projections_df.mean()
    if grouping:
        projections_df = projections_df.groupby(grouping).mean()
    return projections_df.T.corr()



def get_confusion_matrix(projections_df, normalize=True):
    # if projections_df is a string, load the file
    if type(projections_df) == str:
        assert exists(projections_df)
        projections_df = pd.read_json(projections_df)
        
    X = projections_df.iloc[:, 2:]
    y = projections_df.contrast
    clf = LogisticRegressionCV(multi_class='multinomial')
    predict = cross_val_predict(clf, X, y, cv=10)
    cm = confusion_matrix(y, predict)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm
    
                                           

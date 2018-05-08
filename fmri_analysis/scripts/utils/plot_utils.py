from glob import glob
from itertools import chain
import json
from math import ceil
# use backend that doesn't require $DISPLAY environment
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import nilearn.plotting
import nilearn.image
import numpy as np
from os import makedirs
from os.path import basename, join, sep
import pandas as pd
import pickle
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
    
def get_design_df(design_path, subjectinfo_path):
    wf_prefix = design_path.split(sep)[-2].rstrip('_wf')
    designfile_path = join(design_path, 'designfile.mat')
    subjectinfo_path = join(subjectinfo_path, '%s_subjectinfo.pkl' % wf_prefix)
    subjectinfo = pickle.load(open(subjectinfo_path,'rb'))
    desmtx=np.loadtxt(designfile_path,skiprows=5)
    # condition columns and their temporal derivatives
    columns = list(chain(*[[c, c+'_deriv'] for c in subjectinfo.get('conditions', [])]))
    columns += subjectinfo.regressor_names
    design_df = pd.DataFrame(desmtx, columns=columns)
    return design_df

def plot_design(design_df, size=12, output_dir=None):
    if 'junk' in design_df.columns:
        end_index = list(design_df.columns).index('junk')
    else:
        end_index = list(design_df.columns).index('X')
    if end_index != 0:
        max_trial = max(len(design_df)//5, 100)
        regs = design_df.iloc[0:max_trial,0:end_index:2]
        if 'response_time' in design_df.columns:
            regs = pd.concat([regs, design_df.ix[0:max_trial, 'response_time']], axis=1)
        f = plt.figure(figsize=(size,size))
        ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
        ax2 = plt.subplot2grid((3, 2), (1, 0))
        ax3 = plt.subplot2grid((3, 2), (1, 1))
        ax4 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
        # plot regressors over time
        regs.plot(legend=True, ax=ax1, title='TS: Regressors of Interest', fontsize=15)
        ax1.set_xlabel('Trial')
        # plot correlations amongst regressors
        sns.heatmap(regs.corr(), ax=ax2, square=True, annot=True, cbar=False, xticklabels=False)
        ax2.set_title('Heatmap: Regressors of Interest', fontsize=15)
        sns.heatmap(design_df.corr(), ax=ax3, square=True, xticklabels=False)
        ax3.set_title('Heatmap: Design Matrix', fontsize=15)
        #plot design matrix
        sns.heatmap(design_df, ax=ax4, cbar=False)
        ax4.set_title('Design Matrix', fontsize = 15)
        plt.subplots_adjust(wspace=.5, hspace=.5)
    else:
        f = plt.figure(figsize=(12,4))
        sns.heatmap(design_df, cbar=False)
    if output_dir:
        makedirs(output_dir, exist_ok=True)
        f.savefig(join(output_dir,'design_plot.png'))

from nilearn.input_data import NiftiLabelsMasker

def plot_fmri_resid(resid_path, atlas=None):
    image = nilearn.image.load_img(resid_path)
    nilearn.plotting.plot_epi(nilearn.image.mean_img(image))
    if atlas:
        masker = NiftiLabelsMasker(labels_img=atlas['maps'], standardize=True)
        time_series = masker.fit_transform(image)
        time_series = pd.DataFrame(time_series)
        if 'labels' in atlas.keys():
            time_series.columns = atlas['labels']
        # remove ROIs with no variance
        no_variance = time_series.std(axis=0)==0
        print('No variance, removing ROIs: %s' % list(no_variance[no_variance].index))
        time_series = time_series.loc[:,no_variance==False]
        sns.clustermap(time_series.corr(), square=True, figsize=[15,15])
        return time_series
            
        
        

def plot_zmaps(task_path, smoothness=8):
    fmri_contrast_paths = join(task_path, 'zstat?.nii.gz')
    fmri_contrast_files = sorted(glob(fmri_contrast_paths))
    contrasts_path = join(task_path, 'contrasts.pkl')
    contrasts = pickle.load(open(contrasts_path,'rb'))
    contrast_names = [c[0] for c in contrasts]
    for i, contrast_img in enumerate(fmri_contrast_files):
        smooth_img = nilearn.image.smooth_img(contrast_img, smoothness)
        nilearn.plotting.plot_glass_brain(smooth_img,
                                          display_mode='lyrz', 
                                          colorbar=True, 
                                          plot_abs=False, threshold=0,
                                          title=contrast_names[i])

def plot_tstats(task_path, smoothness=8):
    fmri_contrast_paths = join(task_path, 'tstat?.nii.gz')
    fmri_contrast_files = sorted(glob(fmri_contrast_paths))
    contrasts_path = join(task_path, 'contrasts.pkl')
    contrasts = pickle.load(open(contrasts_path,'rb'))
    contrast_names = [c[0] for c in contrasts]
    for i, contrast_img in enumerate(fmri_contrast_files):
        smooth_img = nilearn.image.smooth_img(contrast_img, smoothness)
        nilearn.plotting.plot_stat_map(smooth_img, threshold=0,
                                        title=contrast_names[i])

def plot_contrasts(data_dir, task, plot_individual=False,
               contrast_index=None, output_dir=None):
    """Function to plot contrasts stored in a pickle object
    
    Args:
        data_dir (string): The directory to find the contrast pickle objects
        task (string): the task to pull contrasts for
        smoothness (int): smoothness parameter passed to 
                           nilearn.image_smooth_img. Defaults to 8.
        plot_individual (bool): If true, create plots for individual contrasts
        contrast_index (list): list of contrasts to subselect. If None, use all
        output_dir (string): if specified, save plots here
    """
    # if output_dir is specified, create it to store plots
    if output_dir:
        makedirs(join(output_dir,task), exist_ok=True)
    
    subj_ids = json.load(open(join(data_dir,task,'subj_ids.json'), 'r'))
    contrast_objs = sorted(glob(join(data_dir,task,'*copes.nii.gz')),
                           key = lambda x: '-' in x)
    # get contrast names
    contrast_names = [basename(i).split('_copes')[0][len(task)+1:]
                        for i in contrast_objs]

    # set up subplots for group plots
    group_fig, group_axes = plt.subplots(len(contrast_names), 1,
                                         figsize=(14, 5*len(contrast_names)))
    if len(contrast_names) == 1: group_axes = [group_axes]
    group_fig.suptitle('%s Group Contrasts' % (task[:1].upper() + task[1:])
                       , fontsize=30, 
                       fontweight='bold', y=.97)
    plt.subplots_adjust(top=.95)
    for i,contrast_name in enumerate(contrast_names):
        N = len(subj_ids)
        if contrast_index is not None:
            if i+1 not in contrast_index:
                continue
        if plot_individual == True:
            copes = nilearn.image.iter_img(contrast_objs[i])
            # set up subplots for individual contrasts plots
            contrast_fig, contrast_axes = plt.subplots(ceil(N/2), 2,
                                         figsize=(24, 5*ceil(N/2)),
                                         squeeze=True)
            plt.subplots_adjust(top=.95)
            for img_i, img in enumerate(copes):
                # if plotting individuals, add to individual subplot
                nilearn.plotting.plot_glass_brain(img,
                                              display_mode='lyrz', 
                                              colorbar=True, 
                                              plot_abs=False,
                                              title=subj_ids[img_i],
                                              axes=contrast_fig.axes[img_i])
        if plot_individual and output_dir != None:
            contrast_fig.savefig(join(output_dir,task,
                                   'ind_contrast:%s_%s.png' 
                                   % (task, contrast_name)))
        nilearn.plotting.plot_glass_brain(
                                      nilearn.image.mean_img(contrast_objs[i]),
                                      display_mode='lyrz', 
                                      colorbar=True, 
                                      plot_abs=False,
                                      title=contrast_name+', N: %s' % N,
                                      axes=group_axes[i])
    if output_dir:
        group_fig.savefig(join(output_dir,task,'group_contrasts.png'))
        
def dendroheatmap_left(df, labels = True, label_fontsize = 'large'):
    """
    :df: plot hierarchical clustering and heatmap, dendrogram on left
    """
    #clustering
    corr_vec = 1-df.values[np.triu_indices_from(df,k=1)]
    row_clusters = linkage(corr_vec, method='ward', metric='euclidean') 
    #dendrogram
    row_dendr = dendrogram(row_clusters, labels=df.columns, no_plot = True)
    df_rowclust = df.iloc[row_dendr['leaves'],row_dendr['leaves']]
    sns.set_style("white")
    fig = plt.figure(figsize = [16,16])
    ax = fig.add_axes([.16,.3,.62,.62]) 
    cax = fig.add_axes([0.21,0.25,0.5,0.02]) 
    sns.heatmap(df_rowclust, ax = ax, cbar_ax = cax, cbar_kws = {'orientation': 'horizontal'}, xticklabels = False)
    ax.yaxis.tick_right()
    ax.set_yticklabels(df_rowclust.columns[::-1], rotation=0, rotation_mode="anchor", fontsize = label_fontsize, visible = labels)
    ax.set_xticklabels(df_rowclust.columns, rotation=-90, rotation_mode = "anchor", ha = 'left')
    ax1 = fig.add_axes([.01,.3,.15,.62])
    plt.axis('off')
    row_dendr = dendrogram(row_clusters, orientation='left', ax = ax1) 
    ax1.invert_yaxis()
    return fig, row_dendr['leaves']
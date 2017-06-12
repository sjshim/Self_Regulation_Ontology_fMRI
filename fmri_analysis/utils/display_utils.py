%pylab inline
from glob import glob
from itertools import chain
from matplotlib import pyplot as plt
import nilearn.plotting
import nilearn.image
import numpy as np
from os.path import join
import pandas as pd
import pickle
import seaborn as sns

def get_design_df(task_path):
    designfile_path = join(task_path, 'designfile.mat')
    subjectinfo_path = join(task_path, 'subjectinfo.pkl')
    subjectinfo = pickle.load(open(subjectinfo_path,'rb'))
    desmtx=np.loadtxt(designfile_path,skiprows=5)
    # condition columns and their temporal derivatives
    columns = list(chain(*[[c, c+'_deriv'] for c in subjectinfo.conditions]))
    columns += subjectinfo.regressor_names
    design_df = pd.DataFrame(desmtx, columns=columns)
    return design_df

def plot_design(design_df):
    junk_index = list(design_df.columns).index('junk')
    quintile1 = len(design_df)//5
    regs = design_df.iloc[0:quintile1,0:junk_index:2]
    f, [ax1,ax2,ax3] = plt.subplots(3, 1, figsize=[12,24])
    regs.plot(legend=True, ax=ax1, title='TS: Regressors of Interest')
    sns.heatmap(regs.corr(), ax=ax2, square=True, annot=True, cbar=False)
    ax2.set_title('Heatmap: Regressors of Interest', fontsize=20)
    sns.heatmap(design_df.corr(), ax=ax3, square=True)
    ax3.set_title('Heatmap: Design Matrix', fontsize=20)

def plot_zmaps(task_path, smoothness=8):
    fmri_contrast_paths = join(task_path, 'zstat?.nii.gz')
    fmri_contrast_files = sort(glob(fmri_contrast_paths))
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
    fmri_contrast_files = sort(glob(fmri_contrast_paths))
    contrasts_path = join(task_path, 'contrasts.pkl')
    contrasts = pickle.load(open(contrasts_path,'rb'))
    contrast_names = [c[0] for c in contrasts]
    for i, contrast_img in enumerate(fmri_contrast_files):
        smooth_img = nilearn.image.smooth_img(contrast_img, smoothness)
        nilearn.plotting.plot_stat_map(smooth_img, threshold=0,
                                        title=contrast_names[i])

def group_plot(data_dir, task, smoothness=8, plot_individual=False):
    # get contrast names
    contrast_path = glob(join(data_dir,'*%s/contrasts.pkl' % task))[0]
    contrasts = pickle.load(open(contrast_path,'rb'))
    contrast_names = [c[0] for c in contrasts]
    for i,contrast_name in enumerate(contrast_names):
        map_files = glob(join(data_dir,'*%s/tstat%s.nii.gz' % (task, i+1)))
        smooth_copes = []
        for img in map_files:
            smooth_cope = nilearn.image.smooth_img(img, 8)
            smooth_copes.append(smooth_cope)
            if plot_individual == True:
                nilearn.plotting.plot_glass_brain(smooth_cope,
                                              display_mode='lyrz', 
                                              colorbar=True, 
                                              plot_abs=False)
        nilearn.plotting.plot_glass_brain(nilearn.image.mean_img(smooth_copes),
                                      display_mode='lyrz', 
                                      colorbar=True, 
                                      plot_abs=False,
                                      title=contrast_name)
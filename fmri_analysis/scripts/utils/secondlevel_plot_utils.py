from glob import glob
from os import path, sep

from nilearn import plotting
import matplotlib.pyplot as plt
import seaborn as sns

def plot_2ndlevel_maps(group_path, lookup='*raw*', vmax=None, size=10, threshold=.95):
    task = group_path.split(sep)[-3]
    group_files = sorted(glob(path.join(group_path, lookup)))
    # plot
    f, axes = plt.subplots(len(group_files), 1, figsize=(size, size/2.5*len(group_files)))
    if len(group_files)==1:
        axes = [axes]
    for i, img_path in enumerate(group_files):
        contrast_name = '_'.join(path.basename(img_path).split('_')[:-2])
        if i == 0:
            title = '%s\n%s' % (task, contrast_name)
        else:
            title = contrast_name
        ax = axes[i]
        plotting.plot_glass_brain(img_path,
                                    display_mode='lyrz', 
                                    colorbar=True, vmax=vmax, vmin=vmax,
                                    plot_abs=False, threshold=threshold,
                                    title=title, 
                                    axes=ax)
        
def plot_RDM(RDM, roi=None, title=None, size=8, cluster=True):
    if cluster:
        f = sns.clustermap(RDM, figsize=(size, size))
        if roi:
            ax = f.ax_col_dendrogram
            ax.clear()
            plotting.plot_roi(roi, axes=ax)
        if title:
            f.fig.suptitle(title, fontsize=size*2) 

    else:
        f = plt.figure(figsize=(size, size))
        if roi:
            heatmap_ax = f.add_axes([0,0,.7,.7])
            roi_ax = f.add_axes([0,.75,.7,.2])
            plotting.plot_roi(roi, axes=roi_ax)
        else:
            heatmap_ax = f.add_axes([0,0,1,1])
        sns.heatmap(RDM, ax=heatmap_ax, square=True, xticklabels=[])
        if title:
            heatmap_ax.set_title(title, fontsize=size*2)


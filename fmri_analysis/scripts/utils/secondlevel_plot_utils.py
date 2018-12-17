from glob import glob
from os import path, sep
from nilearn import image, masking, plotting
import matplotlib.pyplot as plt
import seaborn as sns

from utils.plot_utils import save_figure

def plot_2ndlevel_maps(group_path, size=10, threshold=.95, plot_dir=None, ext='png'):
    task = group_path.split(sep)[-3]
    group_files = sorted(glob(path.join(group_path, '*raw*')))
    mask_files = sorted(glob(path.join(group_path, '*corrected*')))
    masks = [image.load_img(img).get_data() > threshold for img in mask_files]
    # plot
    f, axes = plt.subplots(len(group_files), 1, figsize=(size, size/2.5*len(group_files)))
    if len(group_files)==1:
        axes = [axes]
    for i, img_path in enumerate(group_files):
        contrast_name = img_path[img_path.find('_contrast')+1:img_path.find('_file')]
        if i == 0:
            title = '%s: %s' % (task, contrast_name)
        else:
            title = contrast_name
        ax = axes[i]
        # mask image
        mask = image.new_img_like(img_path, masks[i])
        to_plot = image.math_img('img1*mask', img1=img_path, mask=mask)
        plotting.plot_glass_brain(to_plot,
                                    display_mode='lyrz', 
                                    colorbar=True, vmax=None, vmin=None,
                                    plot_abs=False, threshold=threshold,
                                    title=title, 
                                    axes=ax)
    if plot_dir:
        filename = '%s_groupmaps_p<%s.%s' % (task,str(round(1-threshold,2)), ext)
        save_figure(f, path.join(plot_dir, filename))
        
        
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


from matplotlib.colors import ListedColormap
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from nilearn import image, plotting
from nistats.reporting import plot_design_matrix, plot_contrast_matrix

def plot_design(subjinfo, plot_contrasts=False):
    fig, ax = plt.subplots(figsize=(15,8))
    plot_design_matrix(subjinfo.design, ax=ax, rescale=True)
    if plot_contrasts:
        for name, contrast in subjinfo.contrasts:
            ax=plot_contrast_matrix(contrast, design_matrix=subjinfo.design)
            ax.set_xlabel(name)

def plot_contrast(subjinfo, contrast, simple_plot=True, **kwargs):
    if type(contrast) == int:
        contrast = subjinfo.contrasts[contrast]
        contrast_title = contrast[0]
        z_map = subjinfo.fit_model.compute_contrast(contrast[1])
    else:
        contrast_title = contrast
        z_map = subjinfo.fit_model.compute_contrast(contrast)
    plot_map(z_map, title=contrast_title, **kwargs)

def plot_map(contrast_map, title=None, glass_kwargs=None, stat_kwargs=None):
    if glass_kwargs is None:
        glass_kwargs = {}
    if stat_kwargs is None:
        stat_kwargs = {}
    # set up plot
    f, axes = plt.subplots(4, 1, figsize=(12,12))
    # plot glass brain
    glass_args = {'threshold': norm.isf(0.001), 
                 'display_mode': 'ortho'}
    glass_args.update(**glass_kwargs)
    f = plotting.plot_glass_brain(contrast_map, colorbar=True, 
                              title=title, axes=axes[0],
                              plot_abs=False, **glass_args)
    # plot more indepth stats brain
    stat_args = {'threshold': norm.isf(0.001),
                 'cut_coords': 5,
                 'black_bg': True}
    stat_args.update(**stat_kwargs)
    
    plotting.plot_stat_map(contrast_map, display_mode='x', 
                           axes=axes[1], **stat_args)
    plotting.plot_stat_map(contrast_map, display_mode='y', axes=axes[2], **stat_args)                
    plotting.plot_stat_map(contrast_map, display_mode='z', axes=axes[3], **stat_args)
    plt.subplots_adjust(hspace=0)
    return f

def plot_design_timeseries(subjinfo, begin=0, end=-1):
    X_loc = subjinfo.design.columns.get_loc('trans_x')
    subset = subjinfo.design.loc[:, subjinfo.design.columns[:X_loc]]
    subset = subset.drop(columns=subset.filter(regex='_TD').columns)
    for i, col in enumerate(subset.columns):
        subset.loc[:, col] += i*3
    color_palette = sns.color_palette("cubehelix", subset.shape[1])[::-1]
    ax = subset.iloc[begin:end].plot(figsize=(12,10), colormap=ListedColormap(color_palette))
    xlim = ax.get_xlim()
    end = xlim[-1] + (xlim[-1]-xlim[0])*.025
    for i, col in enumerate(subset.columns):
        txt = ax.text(end, i*3, col, color=color_palette[i], fontsize=20)
        txt.set_path_effects([PathEffects.withStroke(linewidth=8, foreground='w')])

def plot_design_heatmap(subjinfo):
    X_loc = subjinfo.design.columns.get_loc('trans_x')
    subset = subjinfo.design.loc[:, subjinfo.design.columns[:X_loc]]
    plt.figure(figsize=(14,14))
    sns.heatmap(subset.corr(), square=True, annot=True, annot_kws={'fontsize': 16})
        
def plot_average_maps(subjects, contrast_keys=None, **kwargs):
    if contrast_keys is None:
        map_keys = subjects[0].maps.keys()
    else:
        map_keys = contrast_keys
    averages = {}
    for key in map_keys:
        try:
            maps = [i.maps[key] for i in subjects]
        except KeyError:
            maps = [i.fit_model.compute_contrast(key) for i in subjects]
        averages[key] = image.mean_img(maps)
    # plot
    for name, average in averages.items():
        default_args = {'threshold': norm.isf(0.001), 
                        'display_mode': 'ortho'}
        default_args.update(**kwargs)
        plotting.plot_glass_brain(average, colorbar=True, 
                              title=name,
                              plot_abs=False, **default_args)
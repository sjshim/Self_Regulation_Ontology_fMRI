import matplotlib.pyplot as plt
import seaborn as sns
from nilearn import plotting

def plot_RDM(RDM, roi=None, title=None, size=8):
    f = plt.figure(figsize=(size, size))
    if roi:
        heatmap_ax = f.add_axes([0,0,.7,.7])
        roi_ax = f.add_axes([0,.75,.7,.2])
        plotting.plot_roi(roi, axes=roi_ax)
    else:
        heatmap_ax = f.add_axes([0,0,1,1])
    sns.heatmap(RDM, ax=heatmap_ax, square=True)
    if title:
        heatmap_ax.set_title(title, fontsize=size*2)
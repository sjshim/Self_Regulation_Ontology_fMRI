# Running the traditional fMRI Analyses:

Set singularity_config.txt and singularity_config_2ndlevels.txt to point to the appropriate singularity images. 2 images are currently being used because the 2ndlevels required a new version of nipype for appropriate calls to randomise. The 2ndlevel image's version of nipype had to be further modified to correctly call the f-only flag; see [this git commit](https://github.com/nipy/nipype/commit/fe4cf08856a521159430aefb031fc4db119cd1fb#diff-d61b311bef9026063313dd51b8bf932eaa0d939343503e70048d450c863ab367) which fixes the issue, but unfortunately using this version of nipype would break too many package dependencies.

Run the following bash files in this order. See their contents to follow the batch files they call, and the python scripts those call.

1. `run_1stlevel_analysis.sh` runs the firstlevels for all tasks.

2. `run_2ndlevel_analysis.sh` runs all 2ndlevel models for all tasks. This includes Jeanette Mumfords RT models, which may not be relevant.  
To only run the intercept/mean models, use `run_2ndlevel_mean_analysis.sh`.  
To run only Jeanette's models, use `run_2ndlevel_RT_analysis.sh`.

3. `run_2ndlevel_visualizations.sh` produces thresholded and unthresholded axial plots of every contrast. 

At this point the traditional analyses are complete, and additional analyses can be applied (e.g. see the notebooks referred to in the README at the root of this repo.).

# Other potentially useful files

1. `build_4d_from_1stlevels.batch` concatenantes the 1stlevels into a single files for easy viewing and visual flagging of potential outliers.  

2. `rest_regress_physio.batch` regresses physio from rest.  

3. `rest_parcel.batch` parcellates the rest data (should only be done after regressing out physio)

4. `timeseries_parcel.batch` parcellates the preprocessed task data.

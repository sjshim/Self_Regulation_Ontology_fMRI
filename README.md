# Navigating and Using the Repo

The following is a list of directories and their contents, in order of importance

## `fmri_analysis`: network discovery analysis

`fmri_analysis` contains the main scripts of the repo for running standard contrast analyses.
The main files are all in `scripts` and the `utils` dir within.  
To run these scripts using Sherlock or other computing clusters, view the README located in `Self_Regulation_Ontology_fMRI/fmri_analysis/batch_files/aim1`.

1. `1stlevel_analysis.py` runs 1stlevels with the help of` utils/firstlevel_utils.py`  
2. `2ndlevel_analysis.py` runs 2ndlevels with the help of `utils/firstlevel_utils.py`, `utils.secondlevel_utils.py` and `utils/utils.py` 
3. '3rdlevel_analysis.py' runs 3rdlevels which is a variation of 2ndlevel analysis (due to multiple sessions for same individual) 
4. `Visualizations.py` can be used to run 1st and 2nd level visualizations based on flags passed to it.  

## `behavioral_data_prep`
Contains code for generating events files from task data and transfering them to the BIDS directory containing the scan data.
  
## `docker_files` and `singularity_images`
The former contains requirement files which help create the docker image. The latter contains the singularity image(s) built out of the docker image, which can be used to run the environment on Sherlock, Stanford HPCC.
  
5. `fmri_data_prep`
Contains scripts and atlases to parcellate the data, used with collaborator Mac Shine.
  
6. `fmri_prepoc`
Legacy code.  

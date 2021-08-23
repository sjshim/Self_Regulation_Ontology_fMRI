# Self_Regulation_Ontology - fMRI analysis

### Setting up the docker image
This is the recommended way to use the repo

Run: 
```
docker build --rm -t fmri_env . 
```
That's it!

In docker\_files/run_docker.txt you can find example docker commands.
If you start the notebook version, 
you can access it at the following url:
http://127.0.0.1:8888/lab?

### Setting up python environment

conda create -n SRO python=3.5.3
source activate SRO
pip install -r requirements.txt

# Navigating and Using the Repo

The following is a list of directories and their contents, in order of importance

## `fmri_analysis`: Aim1 fMRI Analysis (TODO: include aim2, aim4)  

`fmri_analysis` contains the main scripts of the repo for running standard contrast analyses.
The main files are all in `scripts` and the `utils` dir within.  
To run these scripts using Sherlock or other computing clusters, view the README located in `Self_Regulation_Ontology_fMRI/fmri_analysis/batch_files/aim1`.

1. `1stlevel_analysis.py` runs 1stlevels with the help of` utils/firstlevel_utils.py`  
2. `2ndlevel_analysis.py` runs 2ndlevels with the help of `utils/firstlevel_utils.py`, `utils.secondlevel_utils.py` and `utils/utils.py`  
3. `Visualizations.py` can be used to run 1st and 2nd level visualizations based on flags passed to it.  

4. Once these analyses are complete, navigate to `Self_Regulation_Onotology_fMRI/fmri_analysis/scripts/notebooks/` and run 2 notebooks (currently run locally with data mounted; see the first cells): 1) `0_2ndlevel_EFA.ipynb` performs exploratory factor analysis on the 2ndlevel contrasts and examines how parcels fall in factor space, and 2) `1_label_and_functional_distances.ipynb` compares the distances between 2ndlevel contrasts in functional and theory-driven spaces against null models.  

## `behavioral_data_prep`
Contains code for generating events files from task data and transfering them to the BIDS directory containing the scan data.
  
## `docker_files` and `singularity_images`
The former contains requirement files which help create the docker image. The latter contains the singularity image(s) built out of the docker image, which can be used to run the environment on Sherlock, Stanford HPCC.
  
## `fmri_experiments`
Contains the code to run the experiments used in aim1 and aim2 of the project.
  
5. `fmri_data_prep`
Contains scripts and atlases to parcellate the data, used with collaborator Mac Shine.
  
6. `fmri_prepoc`
Legacy code.  

# example to run 1st level analysis
docker run -ti --rm -v ~/Experiments/expfactory/Self_Regulation_Ontology_fMRI/fmri_analysis:/SRO -v $HOME/temp:/output -v /mnt/OAK/:/Data sro_fmri python scripts/task_analysis.py /output /Data --participant s358 --tasks stopSignal 

# developer command
docker run -ti --rm -v ~/Experiments/expfactory/Self_Regulation_Ontology_fMRI/fmri_analysis:/home -v $HOME/temp:/output -v /mnt/OAK/:/Data sro_fmri
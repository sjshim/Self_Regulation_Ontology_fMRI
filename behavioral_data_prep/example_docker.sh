
# DATA ANALYSIS SCRIPTS
# enter into docker environment for development
scripts_loc=~/Experiments/Self_Regulation_Ontology_fMRI/behavioral_data_prep
behavioral_data_loc=~/Experiments/Self_Regulation_Ontology_fMRI/behavioral_data

docker run --rm  \
--mount type=bind,src=$scripts_loc,dst=/scripts \
--mount type=bind,src=$behavioral_data_loc,dst=/behavioral_data \
-ti sro_dataprep 

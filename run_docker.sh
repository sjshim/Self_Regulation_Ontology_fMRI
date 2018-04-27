scripts_loc=~/Experiments/expfactory/Self_Regulation_Ontology_fMRI/fmri_analysis/scripts
data_loc=$HOME/tmp/fmri/data/
output=$HOME/tmp/fmri/output
docker run --rm  \
--mount type=bind,src=$scripts_loc,dst=/scripts \
--mount type=bind,src=$data_loc,dst=/data,readonly \
--mount type=bind,src=$output,dst=/output \
-ti fmri_env \
python task_analysis.py /output /Data --participant s358 --tasks stopSignal 

# mount with x11 forwarding...not working
scripts_loc=~/Experiments/expfactory/Self_Regulation_Ontology_fMRI/fmri_analysis/scripts
data_loc=$HOME/tmp/fmri/data/
output=$HOME/tmp/fmri/output
docker run --rm  \
--mount type=bind,src=$scripts_loc,dst=/scripts \
--mount type=bind,src=$data_loc,dst=/data,readonly \
--mount type=bind,src=$output,dst=/output \
--user=$USER \
--env="DISPLAY" \
--volume="/etc/group:/etc/group:ro" \
--volume="/etc/passwd:/etc/passwd:ro" \
--volume="/etc/shadow:/etc/shadow:ro" \
--volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
-ti fmri_env \
python task_analysis.py /output /Data --participant s358 --tasks stopSignal 

# mount notebook
scripts_loc=~/Experiments/expfactory/Self_Regulation_Ontology_fMRI/fmri_analysis/scripts
derivatives_loc=$HOME/tmp/fmri/derivatives/
docker run -it --rm \
--mount type=bind,src=$scripts_loc,dst=/scripts \
--mount type=bind,src=$derivatives_loc/fmriprep/fmriprep,dst=/data \
--mount type=bind,src=$derivatives_loc/1stlevel,dst=/output \
-p 8888:8888 fmri_notebook

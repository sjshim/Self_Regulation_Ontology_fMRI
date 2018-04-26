scripts_loc=~/Experiments/expfactory/Self_Regulation_Ontology_fMRI/fmri_analysis/scripts
data_loc=$HOME/tmp/fmri/data/
output=$HOME/tmp/fmri/output
docker run --rm  \
--mount type=bind,src=$scripts_loc,dst=/scripts \
--mount type=bind,src=$data_loc,dst=/data,readonly \
--mount type=bind,src=$output,dst=/output \
-ti fmri_env \
python task_analysis.py /output /Data --participant s358 --tasks stopSignal 

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
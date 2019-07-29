derivatives_loc=`sed '6q;d' singularity_config.txt`
task=$1
for subj_path in $derivatives_loc/1stLevel/s???
do
    mv $subj_path/${task}_model-rt_wf-contrast/* $subj_path/${task}/model-rt/wf-contrast/
    mv $subj_path/${task}_model-nort_wf-contrast/* $subj_path/${task}/model-nort/wf-contrast/
    rm -r $subj_path/${task}_model-rt_wf-contrast
    rm -r $subj_path/${task}_model-nort_wf-contrast
done
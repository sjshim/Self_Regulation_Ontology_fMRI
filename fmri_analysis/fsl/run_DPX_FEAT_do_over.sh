fsl_dir="/home/groups/russpold/uh2_analysis/Self_Regulation_Ontology_fMRI/fmri_analysis/fsl"
template_dir=$fsl_dir/"templates"
tmp_dir=$fsl_dir/"tmp_batch"


OLDIFS=$IFS; IFS=',';
for i in /sub-s471/ses-2/func/sub-s471_ses-2_task-DPX_run-1_bold,s471,brain1,1088,696320000 /sub-s512/ses-1/func/sub-s512_ses-1_task-DPX_run-1_bold,s512,brain1,1104,706560000 /sub-s573/ses-1/func/sub-s573_ses-1_task-DPX_run-1_bold,s573,brain1,1102,705280000 /sub-s581/ses-1/func/sub-s581_ses-1_task-DPX_run-1_bold,s581,brain2,1104,706560000 /sub-s611/ses-2/func/sub-s611_ses-2_task-DPX_run-1_bold,s611,brain2,1104,706560000 /sub-s615/ses-1/func/sub-s615_ses-1_task-DPX_run-1_bold,s615,brain1,1104,706560000 /sub-s624/ses-1/func/sub-s624_ses-1_task-DPX_run-1_bold,s624,brain1,1104,706560000 /sub-s626/ses-2/func/sub-s626_ses-2_task-DPX_run-1_bold,s626,brain2,1104,706560000 /sub-s627/ses-2/func/sub-s627_ses-2_task-DPX_run-1_bold,s627,brain2,1104,706560000 /sub-s635/ses-1/func/sub-s635_ses-1_task-DPX_run-1_bold,s635,brain1,1104,706560000 /sub-s638/ses-2/func/sub-s638_ses-2_task-DPX_run-1_bold,s638,brain2,1104,706560000 /sub-s640/ses-2/func/sub-s640_ses-2_task-DPX_run-1_bold,s640,brain2,1091,698240000 /sub-s644/ses-1/func/sub-s644_ses-1_task-DPX_run-1_bold,s644,brain1,1104,706560000 /sub-s649/ses-1/func/sub-s649_ses-1_task-DPX_run-1_bold,s649,brain1,1104,706560000 /sub-s650/ses-1/func/sub-s650_ses-1_task-DPX_run-1_bold,s650,brain1,1104,706560000; do set -- $i;
	sed -e "s|{RELATIVE_BOLD}|$1|g" -e "s|{SUBJECT}|$2|g" -e "s|{SES_BRAIN}|$3|g" -e "s|{NTP}|$4|g" -e "s|{TOT_VOX}|$5|g" $template_dir/template_DPX_RT-True_fsl.fsf > $tmp_dir/DPX_$2_RT-True_fsl.fsf;
    sed -e "s|{RELATIVE_BOLD}|$1|g" -e "s|{SUBJECT}|$2|g" -e "s|{SES_BRAIN}|$3|g" -e "s|{NTP}|$4|g" -e "s|{TOT_VOX}|$5|g" $template_dir/template_DPX_RT-False_fsl.fsf > $tmp_dir/DPX_$2_RT-False_fsl.fsf;
    sed -e "s|{SUBJECT}|$2|g" -e "s|{TASK}|DPX|g" $template_dir/template_1stlevel_FEAT.batch > $tmp_dir/DPX_$2_FEAT.batch;
    sbatch $tmp_dir/DPX_$2_FEAT.batch;
done;
IFS=$OLDIFS;
    
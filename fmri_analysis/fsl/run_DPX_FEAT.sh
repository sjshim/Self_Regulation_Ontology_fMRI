#!/bin/bash
#SBATCH --job-name=DPX_FEAT
#SBATCH --output=.out/DPX_FEAT.out
#SBATCH --error=.err/DPX_FEAT.err
#SBATCH --time=10:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=poldracklab@stanford.edu
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH -p russpold
export PYTHONPATH=""

fsl_dir="/home/groups/russpold/uh2_analysis/Self_Regulation_Ontology_fMRI/fmri_analysis/fsl"
template_dir = $fsl_dir/"templates"
tmp_dir=$fsl_dir/"tmp_batch"




    for i in "$template_dir/template_DPX_fsl.fsf"; do 
     sed -e "s|{RELATIVE_BOLD}|/sub-s061/ses-2/func/sub-s061_ses-2_task-DPX_run-1_bold|g" -e "s|{SUBJECT}|s061|g" -e "s|{SES_BRAIN}|brain1|g" -e "s|{NTP}|1094|g" $template_dir/template_DPX_fsl.fsf > $tmp_dir/DPX_s061_fsl.fsf
     sed -e "s|{SUBJECT}|s061|g" $template_dir/template_fsl_DPX_FEAT.batch > $tmp_dir/DPX_s061_FEAT.batch
     sbatch $tmp_dir/DPX_s061_FEAT.batch
    done  
    

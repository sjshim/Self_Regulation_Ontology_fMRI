fsl_dir="/home/groups/russpold/uh2_analysis/Self_Regulation_Ontology_fMRI/fmri_analysis/fsl"
template_dir=$fsl_dir/"templates"
tmp_dir=$fsl_dir/"tmp_batch"


OLDIFS=$IFS; IFS=',';
for i in /sub-s541/ses-2/func/sub-s541_ses-2_task-DPX_run-1_bold,s541,brain2,1104,706560000 /sub-s548/ses-1/func/sub-s548_ses-1_task-DPX_run-1_bold,s548,brain1,1104,706560000 /sub-s549/ses-1/func/sub-s549_ses-1_task-DPX_run-1_bold,s549,brain1,1104,706560000 /sub-s553/ses-1/func/sub-s553_ses-1_task-DPX_run-1_bold,s553,brain1,1104,706560000 /sub-s554/ses-2/func/sub-s554_ses-2_task-DPX_run-1_bold,s554,brain2,1104,706560000 /sub-s555/ses-2/func/sub-s555_ses-2_task-DPX_run-1_bold,s555,brain1,1104,706560000 /sub-s556/ses-2/func/sub-s556_ses-2_task-DPX_run-1_bold,s556,brain2,1095,700800000 /sub-s557/ses-1/func/sub-s557_ses-1_task-DPX_run-1_bold,s557,brain1,1104,706560000 /sub-s558/ses-1/func/sub-s558_ses-1_task-DPX_run-1_bold,s558,brain1,1104,706560000 /sub-s561/ses-1/func/sub-s561_ses-1_task-DPX_run-1_bold,s561,brain1,1104,706560000 /sub-s567/ses-2/func/sub-s567_ses-2_task-DPX_run-1_bold,s567,brain2,1104,706560000 /sub-s568/ses-2/func/sub-s568_ses-2_task-DPX_run-1_bold,s568,brain2,1093,699520000 /sub-s570/ses-2/func/sub-s570_ses-2_task-DPX_run-1_bold,s570,brain2,1104,706560000 /sub-s572/ses-1/func/sub-s572_ses-1_task-DPX_run-1_bold,s572,brain1,1104,706560000 /sub-s573/ses-1/func/sub-s573_ses-1_task-DPX_run-1_bold,s573,brain1,1102,705280000 /sub-s574/ses-1/func/sub-s574_ses-1_task-DPX_run-1_bold,s574,brain1,1104,706560000 /sub-s577/ses-2/func/sub-s577_ses-2_task-DPX_run-1_bold,s577,brain2,1104,706560000 /sub-s579/ses-2/func/sub-s579_ses-2_task-DPX_run-1_bold,s579,brain2,1104,706560000 /sub-s581/ses-1/func/sub-s581_ses-1_task-DPX_run-1_bold,s581,brain2,1104,706560000 /sub-s582/ses-1/func/sub-s582_ses-1_task-DPX_run-1_bold,s582,brain1,1104,706560000 /sub-s583/ses-1/func/sub-s583_ses-1_task-DPX_run-1_bold,s583,brain1,1104,706560000 /sub-s584/ses-1/func/sub-s584_ses-1_task-DPX_run-1_bold,s584,brain1,1104,706560000 /sub-s585/ses-1/func/sub-s585_ses-1_task-DPX_run-1_bold,s585,brain1,1104,706560000 /sub-s586/ses-1/func/sub-s586_ses-1_task-DPX_run-1_bold,s586,brain1,1104,706560000 /sub-s587/ses-2/func/sub-s587_ses-2_task-DPX_run-1_bold,s587,brain2,1096,701440000 /sub-s588/ses-2/func/sub-s588_ses-2_task-DPX_run-1_bold,s588,brain1,1104,706560000 /sub-s589/ses-2/func/sub-s589_ses-2_task-DPX_run-1_bold,s589,brain2,1090,697600000 /sub-s590/ses-2/func/sub-s590_ses-2_task-DPX_run-1_bold,s590,brain2,1104,706560000 /sub-s591/ses-1/func/sub-s591_ses-1_task-DPX_run-1_bold,s591,brain1,1104,706560000 /sub-s592/ses-1/func/sub-s592_ses-1_task-DPX_run-1_bold,s592,brain1,1088,696320000 /sub-s593/ses-1/func/sub-s593_ses-1_task-DPX_run-1_bold,s593,brain1,1104,706560000 /sub-s594/ses-1/func/sub-s594_ses-1_task-DPX_run-1_bold,s594,brain1,1104,706560000 /sub-s595/ses-1/func/sub-s595_ses-1_task-DPX_run-1_bold,s595,brain1,1104,706560000 /sub-s596/ses-2/func/sub-s596_ses-2_task-DPX_run-1_bold,s596,brain1,1104,706560000 /sub-s597/ses-2/func/sub-s597_ses-2_task-DPX_run-1_bold,s597,brain2,1104,706560000 /sub-s598/ses-2/func/sub-s598_ses-2_task-DPX_run-1_bold,s598,brain1,1104,706560000 /sub-s601/ses-1/func/sub-s601_ses-1_task-DPX_run-1_bold,s601,brain1,1097,702080000 /sub-s602/ses-1/func/sub-s602_ses-1_task-DPX_run-1_bold,s602,brain1,1104,706560000 /sub-s605/ses-2/func/sub-s605_ses-2_task-DPX_run-1_bold,s605,brain2,1104,706560000 /sub-s606/ses-2/func/sub-s606_ses-2_task-DPX_run-1_bold,s606,brain2,1104,706560000 /sub-s607/ses-1/func/sub-s607_ses-1_task-DPX_run-1_bold,s607,brain1,1104,706560000 /sub-s608/ses-1/func/sub-s608_ses-1_task-DPX_run-1_bold,s608,brain1,1104,706560000 /sub-s609/ses-1/func/sub-s609_ses-1_task-DPX_run-1_bold,s609,brain1,1097,702080000 /sub-s610/ses-1/func/sub-s610_ses-1_task-DPX_run-1_bold,s610,brain1,1104,706560000 /sub-s611/ses-2/func/sub-s611_ses-2_task-DPX_run-1_bold,s611,brain2,1104,706560000 /sub-s612/ses-2/func/sub-s612_ses-2_task-DPX_run-1_bold,s612,brain2,1104,706560000 /sub-s613/ses-2/func/sub-s613_ses-2_task-DPX_run-1_bold,s613,brain3,1104,706560000 /sub-s614/ses-2/func/sub-s614_ses-2_task-DPX_run-1_bold,s614,brain2,1104,706560000 /sub-s615/ses-1/func/sub-s615_ses-1_task-DPX_run-1_bold,s615,brain1,1104,706560000 /sub-s616/ses-1/func/sub-s616_ses-1_task-DPX_run-1_bold,s616,brain1,1104,706560000 /sub-s617/ses-1/func/sub-s617_ses-1_task-DPX_run-1_bold,s617,brain1,1104,706560000 /sub-s618/ses-1/func/sub-s618_ses-1_task-DPX_run-1_bold,s618,brain2,1104,706560000 /sub-s619/ses-2/func/sub-s619_ses-2_task-DPX_run-1_bold,s619,brain2,1090,697600000 /sub-s621/ses-2/func/sub-s621_ses-2_task-DPX_run-1_bold,s621,brain2,1104,706560000 /sub-s622/ses-1/func/sub-s622_ses-1_task-DPX_run-1_bold,s622,brain1,1104,706560000 /sub-s623/ses-3/func/sub-s623_ses-3_task-DPX_run-1_bold,s623,brain3,1104,706560000 /sub-s624/ses-1/func/sub-s624_ses-1_task-DPX_run-1_bold,s624,brain1,1104,706560000 /sub-s626/ses-2/func/sub-s626_ses-2_task-DPX_run-1_bold,s626,brain2,1104,706560000 /sub-s627/ses-2/func/sub-s627_ses-2_task-DPX_run-1_bold,s627,brain2,1104,706560000 /sub-s628/ses-2/func/sub-s628_ses-2_task-DPX_run-1_bold,s628,brain2,1104,706560000 /sub-s629/ses-2/func/sub-s629_ses-2_task-DPX_run-1_bold,s629,brain2,1104,706560000 /sub-s631/ses-1/func/sub-s631_ses-1_task-DPX_run-1_bold,s631,brain1,1097,702080000 /sub-s633/ses-1/func/sub-s633_ses-1_task-DPX_run-1_bold,s633,brain1,1098,702720000 /sub-s634/ses-1/func/sub-s634_ses-1_task-DPX_run-1_bold,s634,brain1,1104,706560000 /sub-s635/ses-1/func/sub-s635_ses-1_task-DPX_run-1_bold,s635,brain1,1104,706560000 /sub-s636/ses-2/func/sub-s636_ses-2_task-DPX_run-1_bold,s636,brain2,1104,706560000 /sub-s637/ses-2/func/sub-s637_ses-2_task-DPX_run-1_bold,s637,brain2,1104,706560000 /sub-s638/ses-2/func/sub-s638_ses-2_task-DPX_run-1_bold,s638,brain2,1104,706560000 /sub-s640/ses-2/func/sub-s640_ses-2_task-DPX_run-1_bold,s640,brain2,1091,698240000 /sub-s641/ses-1/func/sub-s641_ses-1_task-DPX_run-1_bold,s641,brain1,1086,695040000 /sub-s642/ses-1/func/sub-s642_ses-1_task-DPX_run-1_bold,s642,brain1,1104,706560000 /sub-s643/ses-1/func/sub-s643_ses-1_task-DPX_run-1_bold,s643,brain1,1104,706560000 /sub-s644/ses-1/func/sub-s644_ses-1_task-DPX_run-1_bold,s644,brain1,1104,706560000 /sub-s645/ses-2/func/sub-s645_ses-2_task-DPX_run-1_bold,s645,brain2,1104,706560000 /sub-s647/ses-2/func/sub-s647_ses-2_task-DPX_run-1_bold,s647,brain2,1104,706560000 /sub-s648/ses-1/func/sub-s648_ses-1_task-DPX_run-1_bold,s648,brain1,1104,706560000 /sub-s649/ses-1/func/sub-s649_ses-1_task-DPX_run-1_bold,s649,brain1,1104,706560000 /sub-s650/ses-1/func/sub-s650_ses-1_task-DPX_run-1_bold,s650,brain1,1104,706560000; do set -- $i;
	sed -e "s|{RELATIVE_BOLD}|$1|g" -e "s|{SUBJECT}|$2|g" -e "s|{SES_BRAIN}|$3|g" -e "s|{NTP}|$4|g" -e "s|{TOT_VOX}|$5|g" $template_dir/template_DPX_RT-True_fsl.fsf > $tmp_dir/DPX_$2_RT-True_fsl.fsf;
    sed -e "s|{RELATIVE_BOLD}|$1|g" -e "s|{SUBJECT}|$2|g" -e "s|{SES_BRAIN}|$3|g" -e "s|{NTP}|$4|g" -e "s|{TOT_VOX}|$5|g" $template_dir/template_DPX_RT-False_fsl.fsf > $tmp_dir/DPX_$2_RT-False_fsl.fsf;
    sed -e "s|{SUBJECT}|$2|g" -e "s|{TASK}|DPX|g" $template_dir/template_1stlevel_FEAT.batch > $tmp_dir/DPX_$2_FEAT.batch;
    sbatch $tmp_dir/DPX_$2_FEAT.batch;
done;
IFS=$OLDIFS;
    

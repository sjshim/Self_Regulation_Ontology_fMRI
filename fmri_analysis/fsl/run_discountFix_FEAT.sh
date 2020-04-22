fsl_dir="/home/groups/russpold/uh2_analysis/Self_Regulation_Ontology_fMRI/fmri_analysis/fsl"
template_dir=$fsl_dir/"templates"
tmp_dir=$fsl_dir/"tmp_batch"


OLDIFS=$IFS; IFS=',';
 for i in /sub-s061/ses-2/func/sub-s061_ses-2_task-discountFix_run-1_bold,s061,brain1,956 /sub-s130/ses-1/func/sub-s130_ses-1_task-discountFix_run-1_bold,s130,brain1,921 /sub-s172/ses-1/func/sub-s172_ses-1_task-discountFix_run-1_bold,s172,brain1,984 /sub-s192/ses-1/func/sub-s192_ses-1_task-discountFix_run-1_bold,s192,brain1,918 /sub-s234/ses-2/func/sub-s234_ses-2_task-discountFix_run-1_bold,s234,brain2,925 /sub-s251/ses-2/func/sub-s251_ses-2_task-discountFix_run-1_bold,s251,brain2,940 /sub-s358/ses-2/func/sub-s358_ses-2_task-discountFix_run-1_bold,s358,brain2,947 /sub-s373/ses-2/func/sub-s373_ses-2_task-discountFix_run-1_bold,s373,brain2,948 /sub-s445/ses-3/func/sub-s445_ses-3_task-discountFix_run-1_bold,s445,brain1,923 /sub-s465/ses-2/func/sub-s465_ses-2_task-discountFix_run-1_bold,s465,brain2,940 /sub-s471/ses-2/func/sub-s471_ses-2_task-discountFix_run-1_bold,s471,brain1,925 /sub-s483/ses-1/func/sub-s483_ses-1_task-discountFix_run-1_bold,s483,brain1,926 /sub-s491/ses-2/func/sub-s491_ses-2_task-discountFix_run-1_bold,s491,brain1,1065 /sub-s495/ses-2/func/sub-s495_ses-2_task-discountFix_run-1_bold,s495,brain1,1065 /sub-s497/ses-1/func/sub-s497_ses-1_task-discountFix_run-1_bold,s497,brain1,933 /sub-s499/ses-2/func/sub-s499_ses-2_task-discountFix_run-1_bold,s499,brain1,182 /sub-s512/ses-1/func/sub-s512_ses-1_task-discountFix_run-1_bold,s512,brain1,938 /sub-s518/ses-1/func/sub-s518_ses-1_task-discountFix_run-1_bold,s518,brain1,924 /sub-s519/ses-2/func/sub-s519_ses-2_task-discountFix_run-1_bold,s519,brain2,933 /sub-s524/ses-1/func/sub-s524_ses-1_task-discountFix_run-1_bold,s524,brain1,940 /sub-s525/ses-1/func/sub-s525_ses-1_task-discountFix_run-1_bold,s525,brain1,940 /sub-s526/ses-1/func/sub-s526_ses-1_task-discountFix_run-1_bold,s526,brain1,926 /sub-s541/ses-2/func/sub-s541_ses-2_task-discountFix_run-1_bold,s541,brain2,928 /sub-s546/ses-2/func/sub-s546_ses-2_task-discountFix_run-1_bold,s546,brain2,962 /sub-s548/ses-1/func/sub-s548_ses-1_task-discountFix_run-1_bold,s548,brain1,930 /sub-s549/ses-1/func/sub-s549_ses-1_task-discountFix_run-1_bold,s549,brain1,933 /sub-s553/ses-1/func/sub-s553_ses-1_task-discountFix_run-1_bold,s553,brain1,964 /sub-s554/ses-2/func/sub-s554_ses-2_task-discountFix_run-1_bold,s554,brain2,940 /sub-s555/ses-2/func/sub-s555_ses-2_task-discountFix_run-1_bold,s555,brain1,952 /sub-s556/ses-2/func/sub-s556_ses-2_task-discountFix_run-1_bold,s556,brain2,924 /sub-s557/ses-1/func/sub-s557_ses-1_task-discountFix_run-1_bold,s557,brain1,925 /sub-s558/ses-1/func/sub-s558_ses-1_task-discountFix_run-1_bold,s558,brain1,984 /sub-s561/ses-1/func/sub-s561_ses-1_task-discountFix_run-1_bold,s561,brain1,984 /sub-s567/ses-2/func/sub-s567_ses-2_task-discountFix_run-1_bold,s567,brain2,918 /sub-s568/ses-2/func/sub-s568_ses-2_task-discountFix_run-1_bold,s568,brain2,911 /sub-s570/ses-2/func/sub-s570_ses-2_task-discountFix_run-1_bold,s570,brain2,957 /sub-s573/ses-1/func/sub-s573_ses-1_task-discountFix_run-1_bold,s573,brain1,918 /sub-s574/ses-1/func/sub-s574_ses-1_task-discountFix_run-1_bold,s574,brain1,925 /sub-s577/ses-2/func/sub-s577_ses-2_task-discountFix_run-1_bold,s577,brain2,933 /sub-s579/ses-2/func/sub-s579_ses-2_task-discountFix_run-1_bold,s579,brain2,984 /sub-s581/ses-1/func/sub-s581_ses-1_task-discountFix_run-1_bold,s581,brain1,918 /sub-s582/ses-1/func/sub-s582_ses-1_task-discountFix_run-1_bold,s582,brain1,944 /sub-s583/ses-1/func/sub-s583_ses-1_task-discountFix_run-1_bold,s583,brain1,980 /sub-s584/ses-1/func/sub-s584_ses-1_task-discountFix_run-1_bold,s584,brain1,947 /sub-s585/ses-1/func/sub-s585_ses-1_task-discountFix_run-1_bold,s585,brain1,984 /sub-s586/ses-1/func/sub-s586_ses-1_task-discountFix_run-1_bold,s586,brain1,933 /sub-s587/ses-2/func/sub-s587_ses-2_task-discountFix_run-1_bold,s587,brain2,933 /sub-s588/ses-2/func/sub-s588_ses-2_task-discountFix_run-1_bold,s588,brain1,925 /sub-s589/ses-2/func/sub-s589_ses-2_task-discountFix_run-1_bold,s589,brain2,918 /sub-s590/ses-2/func/sub-s590_ses-2_task-discountFix_run-1_bold,s590,brain2,932 /sub-s591/ses-1/func/sub-s591_ses-1_task-discountFix_run-1_bold,s591,brain1,918 /sub-s592/ses-1/func/sub-s592_ses-1_task-discountFix_run-1_bold,s592,brain1,929 /sub-s593/ses-1/func/sub-s593_ses-1_task-discountFix_run-1_bold,s593,brain1,916 /sub-s594/ses-1/func/sub-s594_ses-1_task-discountFix_run-1_bold,s594,brain1,933 /sub-s595/ses-1/func/sub-s595_ses-1_task-discountFix_run-1_bold,s595,brain1,918 /sub-s596/ses-2/func/sub-s596_ses-2_task-discountFix_run-1_bold,s596,brain1,928 /sub-s597/ses-2/func/sub-s597_ses-2_task-discountFix_run-1_bold,s597,brain2,955 /sub-s598/ses-2/func/sub-s598_ses-2_task-discountFix_run-1_bold,s598,brain1,925 /sub-s601/ses-1/func/sub-s601_ses-1_task-discountFix_run-1_bold,s601,brain1,933 /sub-s602/ses-1/func/sub-s602_ses-1_task-discountFix_run-1_bold,s602,brain1,947 /sub-s605/ses-2/func/sub-s605_ses-2_task-discountFix_run-1_bold,s605,brain2,925 /sub-s606/ses-2/func/sub-s606_ses-2_task-discountFix_run-1_bold,s606,brain2,970 /sub-s607/ses-1/func/sub-s607_ses-1_task-discountFix_run-1_bold,s607,brain1,940 /sub-s608/ses-1/func/sub-s608_ses-1_task-discountFix_run-1_bold,s608,brain1,928 /sub-s609/ses-1/func/sub-s609_ses-1_task-discountFix_run-1_bold,s609,brain1,925 /sub-s610/ses-1/func/sub-s610_ses-1_task-discountFix_run-1_bold,s610,brain1,943 /sub-s611/ses-2/func/sub-s611_ses-2_task-discountFix_run-1_bold,s611,brain2,984 /sub-s612/ses-2/func/sub-s612_ses-2_task-discountFix_run-1_bold,s612,brain2,962 /sub-s613/ses-2/func/sub-s613_ses-2_task-discountFix_run-1_bold,s613,brain3,894 /sub-s614/ses-2/func/sub-s614_ses-2_task-discountFix_run-1_bold,s614,brain2,945 /sub-s615/ses-1/func/sub-s615_ses-1_task-discountFix_run-1_bold,s615,brain1,984 /sub-s616/ses-1/func/sub-s616_ses-1_task-discountFix_run-1_bold,s616,brain1,967 /sub-s617/ses-1/func/sub-s617_ses-1_task-discountFix_run-1_bold,s617,brain1,984 /sub-s618/ses-2/func/sub-s618_ses-2_task-discountFix_run-1_bold,s618,brain2,955 /sub-s619/ses-2/func/sub-s619_ses-2_task-discountFix_run-1_bold,s619,brain2,962 /sub-s621/ses-2/func/sub-s621_ses-2_task-discountFix_run-1_bold,s621,brain2,947 /sub-s622/ses-1/func/sub-s622_ses-1_task-discountFix_run-1_bold,s622,brain1,938 /sub-s623/ses-1/func/sub-s623_ses-1_task-discountFix_run-1_bold,s623,brain1,937 /sub-s624/ses-1/func/sub-s624_ses-1_task-discountFix_run-1_bold,s624,brain1,933 /sub-s626/ses-2/func/sub-s626_ses-2_task-discountFix_run-1_bold,s626,brain2,923 /sub-s627/ses-2/func/sub-s627_ses-2_task-discountFix_run-1_bold,s627,brain2,924 /sub-s628/ses-2/func/sub-s628_ses-2_task-discountFix_run-1_bold,s628,brain2,919 /sub-s629/ses-2/func/sub-s629_ses-2_task-discountFix_run-1_bold,s629,brain2,931 /sub-s631/ses-1/func/sub-s631_ses-1_task-discountFix_run-1_bold,s631,brain1,925 /sub-s633/ses-1/func/sub-s633_ses-1_task-discountFix_run-1_bold,s633,brain1,984 /sub-s634/ses-1/func/sub-s634_ses-1_task-discountFix_run-1_bold,s634,brain1,984 /sub-s635/ses-1/func/sub-s635_ses-1_task-discountFix_run-1_bold,s635,brain1,940 /sub-s636/ses-2/func/sub-s636_ses-2_task-discountFix_run-1_bold,s636,brain2,941 /sub-s637/ses-2/func/sub-s637_ses-2_task-discountFix_run-1_bold,s637,brain2,927 /sub-s638/ses-2/func/sub-s638_ses-2_task-discountFix_run-1_bold,s638,brain2,930 /sub-s640/ses-2/func/sub-s640_ses-2_task-discountFix_run-1_bold,s640,brain2,925 /sub-s641/ses-1/func/sub-s641_ses-1_task-discountFix_run-1_bold,s641,brain1,933 /sub-s642/ses-1/func/sub-s642_ses-1_task-discountFix_run-1_bold,s642,brain1,924 /sub-s643/ses-1/func/sub-s643_ses-1_task-discountFix_run-1_bold,s643,brain1,922 /sub-s644/ses-1/func/sub-s644_ses-1_task-discountFix_run-1_bold,s644,brain1,944 /sub-s645/ses-2/func/sub-s645_ses-2_task-discountFix_run-1_bold,s645,brain2,984 /sub-s648/ses-1/func/sub-s648_ses-1_task-discountFix_run-1_bold,s648,brain1,954 /sub-s649/ses-1/func/sub-s649_ses-1_task-discountFix_run-1_bold,s649,brain1,339 /sub-s650/ses-1/func/sub-s650_ses-1_task-discountFix_run-1_bold,s650,brain1,984; do set -- $i;
	sed -e "s|{RELATIVE_BOLD}|$1|g" -e "s|{SUBJECT}|$2|g" -e "s|{SES_BRAIN}|$3|g" -e "s|{NTP}|$4|g" $template_dir/template_DPX_fsl.fsf > $tmp_dir/DPX_$2_fsl.fsf;
    	sed -e "s|{SUBJECT}|$2|g" $template_dir/template_fsl_DPX_FEAT.batch > $tmp_dir/DPX_$2_FEAT.batch;
    	sbatch $tmp_dir/DPX_$2_FEAT.batch;
    done;
    IFS=$OLDIFS;
    
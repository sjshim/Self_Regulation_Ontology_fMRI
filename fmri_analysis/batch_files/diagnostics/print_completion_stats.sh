derivatives_loc=`sed '6q;d' ../singularity_config.txt`
for task in ANT CCTHot discountFix DPX motorSelectiveStop stopSignal stroop surveyMedley twoByTwo WATT3
do
    echo "***************************************************************"
    echo $task
    rtcount=`ls ${derivatives_loc}/1stLevel/*/$task/*-rt/*/cope1.nii* | wc -l`
    nortcount=`ls ${derivatives_loc}/1stLevel/*/$task/*-nort/*/cope1.nii* | wc -l`
    echo "    contrast RT count: $rtcount"
    echo "    contrast noRT count: $nortcount"
done

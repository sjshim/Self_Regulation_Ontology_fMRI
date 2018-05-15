derivatives_loc=`sed '6q;d' singularity_config.txt`
data_loc=`sed '8q;d' singularity_config.txt`
analysis_flag=""
if [ $2 == "beta" ]; then
    analysis_flag="--skip_contrast"
elif [ $2 == "contrast" ]; then
    analysis_flag="--skip_beta"
fi

for path in ${derivatives_loc}/fmriprep/fmriprep/sub-$1
do
    sid=${path: -4}
    echo ""
    echo ******************************${sid}***************************************
    echo ""
    tasks=""
    # find tasks that haven't been run...
    for task in ANT CCTHot discountFix DPX motorSelectiveStop stopSignal stroop surveyMedley twoByTwo WATT3
    #for task in DPX
    do
        # ...with RT as a regressor
        if [ -f ${derivatives_loc}/1stlevel/${sid}/${task}/model-nort/wf-$2/cope1.nii.gz ]; 
        then
            : # echo task analysis already run on $sid $task
        else
            if [ -f ${data_loc}/*${sid}/*/func/*${task}*events.tsv -a -f ${derivatives_loc}/fmriprep/fmriprep/*${sid}/*/func/*${task}*confounds.tsv -a -f ${derivatives_loc}/fmriprep/fmriprep/*${sid}/*/func/*${task}*MNI*preproc.nii.gz ];  then
                tasks="$tasks$task "
            fi
        fi

    done
    if [ "$tasks" != "" ]; then
        echo Running $sid task analysis on $tasks
        sed -e "s/{sid}/$sid/g" -e "s/{tasks}/$tasks/g" -e "s/{ANALYSIS_FLAG}/$analysis_flag/g"  1stlevel_analysis.batch | sbatch
    fi

done

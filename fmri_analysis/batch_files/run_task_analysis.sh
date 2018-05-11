derivatives_loc=`sed '6q;d' singularity_config.txt`
data_loc=`sed '8q;d' singularity_config.txt`
analysis_flag=""
if [ $1 == "beta" ]; then
    analysis_flag="--skip_contrast"
elif [ $1 == "contrast" ]; then
    analysis_flag="--skip_beta"
fi


for path in ${derivatives_loc}/fmriprep/fmriprep/sub-s6??
do
    sid=${path: -4}
    echo ""
    echo ******************************${sid}***************************************
    echo ""
    tasks=""
    # find tasks that haven't been run...
    for task in ANT CCTHot discountFix DPX motorSelectiveStop stopSignal stroop surveyMedley twoByTwo WATT3
    do
        # ...with RT as a regressor
        if [ -f ${derivatives_loc}/1stLevel/${sid}/${task}/model-rt/wf-contrast/cope1.nii.gz ]; 
        then
            : # echo task analysis already run on $sid $task
        else
            if [ -f ${data_loc}/*${sid}/*/func/*${task}*events.tsv -a -f ${derivatives_loc}/fmriprep/fmriprep/*${sid}/*/func/*${task}*confounds.tsv ];  then
                tasks="$tasks$task "
            fi
        fi

    done
    if [ "$tasks" != "" ]; then
        echo Running $sid task analysis on $tasks
        sed -e "s/{sid}/$sid/g" -e "s/{tasks}/$tasks/g" -e "s/{ANALYSIS_FLAG}/$analysis_flag/g" task_analysis.batch | sbatch 
    fi

done

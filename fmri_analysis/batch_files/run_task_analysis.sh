analysis_loc=`sed '6q;d' singularity_config.txt`

for path in ${analysis_loc}/fmriprep/fmriprep/sub-s???
do
    sid=${path: -4}
    echo ""
    echo ******************************${sid}***************************************
    echo ""
    tasks=""
    tasks_noRT=""
    # find tasks that haven't been run...
    for task in ANT CCTHot discountFix DPX motorSelectiveStop stopSignal stroop surveyMedley twoByTwo WATT3
    do
        # ...with RT as a regressor
        if [ -f ${analysis_loc}/output/1stLevel/${sid}_task_${task}/cope1.nii.gz ]; 
        then
            : # echo task analysis already run on $sid $task
        else
            if [ -f ${analysis_loc}/fmriprep/fmriprep/*${sid}/*/func/*${task}*events.tsv -a -f ${analysis_loc}/fmriprep/fmriprep/*${sid}/*/func/*${task}*confounds.tsv ];  then
                tasks="$tasks$task "
            fi
        fi
        # ...without RT as a regressor
        if [ -f ${analysis_loc}/output_noRT/1stLevel/${sid}_task_${task}/cope1.nii.gz ]; 
        then
            : # echo task analysis noRT already run on $sid $task
        else
            if [ -f ${analysis_loc}/fmriprep/fmriprep/*${sid}/*/func/*${task}*events.tsv -a -f ${analysis_loc}/fmriprep/fmriprep/*${sid}/*/func/*${task}*confounds.tsv ]; then
                tasks_noRT="$tasks_noRT$task "
            fi
        fi
    done
    if [ "$tasks" != "" ]; then
        echo Running $sid task analysis on $tasks
        sed -e "s/{sid}/$sid/g" -e "s/{tasks}/$tasks/g" task_analysis.batch | sbatch --time=20:00:00 #-p russpold
    fi

    if [ "$tasks_noRT" != "" ]; then
        echo Running $sid task analysis noRT on $tasks_noRT
        sed -e "s/{sid}/$sid/g" -e "s/{tasks}/$tasks_noRT/g" task_analysis_noRT.batch | sbatch --time=20:00:00 #-p russpold
    fi

done

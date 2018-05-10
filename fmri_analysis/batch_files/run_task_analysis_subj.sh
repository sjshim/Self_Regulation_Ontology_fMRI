derivatives_loc=`sed '6q;d' singularity_config.txt`
data_loc=`sed '8q;d' singularity_config.txt`

for path in ${derivatives_loc}/fmriprep/fmriprep/sub-$1
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
        if [ -f ${derivatives_loc}/1stLevel/${sid}/${task}/model-rt/contrast_wf/cope1.nii.gz ]; 
        then
            : # echo task analysis already run on $sid $task
        else
            if [ -f ${data_loc}/*${sid}/*/func/*${task}*events.tsv -a -f ${derivatives_loc}/fmriprep/fmriprep/*${sid}/*/func/*${task}*confounds.tsv ];  then
                tasks="$tasks$task "
            fi
        fi
        # ...with RT as a regressor
        if [ -f ${derivatives_loc}/1stLevel/${sid}/${task}/model-nort/contrast_wf/cope1.nii.gz ];
        then
            : # echo task analysis already run on $sid $task
        else
            if [ -f ${data_loc}/*${sid}/*/func/*${task}*events.tsv -a -f ${derivatives_loc}/fmriprep/fmriprep/*${sid}/*/func/*${task}*confounds.tsv ];  then
                tasks_noRT="$tasks$task "
            fi
        fi

    done
    if [ "$tasks" != "" ]; then
        echo Running $sid task analysis on $tasks
        sed -e "s/{sid}/$sid/g" -e "s/{tasks}/$tasks/g" -e "s/{RT_flag}//g" task_analysis.batch | sbatch
    fi
    if [ "$tasks_noRT" != "" ]; then
        echo Running $sid task analysis without RT on $tasks_noRT
        sed -e "s/{sid}/$sid/g" -e "s/{tasks}/$tasks_noRT/g" -e "s/{RT_flag}/--ignore_rt/g" task_analysis.batch | sbatch 
    fi
done

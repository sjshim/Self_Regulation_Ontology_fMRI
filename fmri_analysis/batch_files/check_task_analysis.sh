for path in /scratch/PI/russpold/work/ieisenbe/uh2/fmriprep/fmriprep/sub-s???
do
    sid=${path: -4}
    echo ""
    echo ******************************${sid}***************************************
    echo ""
    task_count=0
    task_count_noRT=0
    runtasks=""
    runtasks_noRT=""
    tasks=""
    tasks_noRT=""
    # find tasks that haven't been run...
    for task in ANT CCTHot discountFix DPX motorSelectiveStop stopSignal stroop surveyMedley twoByTwo WATT3
    do
        # ...with RT as a regressor
        if [ -f /scratch/PI/russpold/work/ieisenbe/uh2/output/1stLevel/${sid}_task_${task}/cope1.nii.gz ]; 
        then
           ((task_count+=1))
            runtasks="$runtasks$task "
        else
            if [ -f /scratch/PI/russpold/work/ieisenbe/uh2/fmriprep/fmriprep/*${sid}/*/func/*${task}*events.tsv -a -f /scratch/PI/russpold/work/ieisenbe/uh2/fmriprep/fmriprep/*${sid}/*/func/*${task}*confounds.tsv ];  then
                tasks="$tasks$task "
            fi
        fi
        # ...without RT as a regressor
        if [ -f /scratch/PI/russpold/work/ieisenbe/uh2/output_noRT/1stLevel/${sid}_task_${task}/cope1.nii.gz ]; 
        then
            ((task_count_noRT+=1))
            runtasks_noRT="$runtasks_noRT$task "
        else
            if [ -f /scratch/PI/russpold/work/ieisenbe/uh2/fmriprep/fmriprep/*${sid}/*/func/*${task}*events.tsv -a -f /scratch/PI/russpold/work/ieisenbe/uh2/fmriprep/fmriprep/*${sid}/*/func/*${task}*confounds.tsv ]; then
                tasks_noRT="$tasks_noRT$task "
            fi
        fi
    done
    echo Tasks Run: $task_count, Tasks Run noRT: $task_count_noRT
    echo $sid already run on $runtasks
    if [ "$tasks" != "" ]; then
        echo "** Running $sid task analysis on $tasks"
    fi
    
    echo $sid already run on $runtasks_noRT
    if [ "$tasks_noRT" != "" ]; then
        echo "** Running $sid task analysis noRT on $tasks_noRT"
    fi

done

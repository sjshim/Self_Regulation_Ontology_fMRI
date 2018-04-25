set -e
mriqc_path=`sed '2q;d' singularity_config.txt`
data_path=`sed '6q;d' singularity_config.txt`
out_path=`sed '8q;d' singularity_config.txt`

ignore_list=ignore_list.txt
subjects_run=0
subjects_completed=0
for path in ${data_path}/sub*
do  
    sid=${path:(-4)}
    echo "*******************${sid}************************"
    if grep -Fxq "$sid" $ignore_list; then
        echo $sid is being ignored
    else
        check=0
        if [[  -d ${data_path}/sub-${sid}/ses-1 ]]; then
            files=( $(find ${out_path}/mriqc/reports/ -name "*${sid}*ses-1*run-*") )
            if [[ ${#files[@]} -ne 0 ]]; then
                echo mriqc session 1 run
            else
                check+=1
            fi
        fi
        if [[  -d ${data_path}/sub-${sid}/ses-2 ]]; then
            files=( $(find ${out_path}/mriqc/reports/ -name "*${sid}*ses-2*run-*") )
            if [[ ${#files[@]} -ne 0 ]]; then
                echo mriqc session 2 run
            else
                check+=1
            fi
        fi
        if [[  -d ${data_path}/sub-${sid}/ses-3 ]]; then
            files=( $(find ${out_path}/mriqc/reports/ -name "*${sid}*ses-3*run-*") )
            if [[ ${#files[@]} -ne 0 ]]; then
                echo mriqc session 3 run
            else
                check+=1
            fi
        fi
        if [[ $check>0 ]]; then
            echo Running MRIQC on $sid
            echo singularity run ${mriqc_path} ${data_path} ${out_path}/mriqc participant --participant_label $sid -w $LOCAL_SCRATCH --ants-nthreads 8 --n_procs 16 --mem_gb 110 --verbose-reports >> preproc_task_list.sh
            (( subjects_run+=1 ))
        else
            (( subjects_completed+=1 ))
        fi

    fi
done
echo Subjects running: $subjects_run, subjects completed: $subjects_completed

set -e
ignore_list=ignore_list.txt
for path in data/sub*
do
    echo "*******************************************"
    sid=${path:(-4)}
    if grep -Fxq "$sid" $ignore_list; then
        echo $sid is being ignored
    else
        if [[ -f /scratch/PI/russpold/work/ieisenbe/uh2/mriqc/reports/sub-${sid}_ses-1_task-rest_run-1_bold.html && 
            -f /scratch/PI/russpold/work/ieisenbe/uh2/mriqc/reports/sub-${sid}_ses-2_task-rest_run-1_bold.html ]]; then
            echo MRIQC already run on $sid
        else
            echo Running MRIQC on $sid
            sed "s/{sid}/$sid/g" mriqc.batch | sbatch -p russpold
        fi
    fi
done


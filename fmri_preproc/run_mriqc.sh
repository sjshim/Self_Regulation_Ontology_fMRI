set -e
ignore_list=ignore_list.txt
subjects_run=0
subjects_completed=0

for path in /oak/stanford/groups/russpold/data/uh2/sub*
do  
    sid=${path:(-4)}
    echo "*******************${sid}************************"
    if grep -Fxq "$sid" $ignore_list; then
        echo $sid is being ignored
    else
        check=0
        if [[  -d /oak/stanford/groups/russpold/data/uh2/sub-${sid}/ses-1 ]]; then
            files=( $(find /scratch/PI/russpold/work/ieisenbe/uh2/mriqc/reports/ -name "*${sid}*ses-1*run-*") )
            if [[ ${#files[@]} -ne 0 ]]; then
                echo mriqc session 1 run
            else
                check+=1
            fi
        fi
        if [[  -d /oak/stanford/groups/russpold/data/uh2/sub-${sid}/ses-2 ]]; then
            files=( $(find /scratch/PI/russpold/work/ieisenbe/uh2/mriqc/reports/ -name "*${sid}*ses-2*run-*") )
            if [[ ${#files[@]} -ne 0 ]]; then
                echo mriqc session 2 run
            else
                check+=1
            fi
        fi
        if [[  -d /oak/stanford/groups/russpold/data/uh2/sub-${sid}/ses-3 ]]; then
            files=( $(find /scratch/PI/russpold/work/ieisenbe/uh2/mriqc/reports/ -name "*${sid}*ses-3*run-*") )
            if [[ ${#files[@]} -ne 0 ]]; then
                echo mriqc session 3 run
            else
                check+=1
            fi
        fi
        if [[ $check>0 ]]; then
            echo Running MRIQC on $sid
            sed "s/{sid}/$sid/g" mriqc.batch | sbatch -p russpold
            (( subjects_run+=1 ))
        else
            (( subjects_completed+=1 ))
        fi

    fi
done
echo Subjects running: $subjects_run, subjects completed: $subjects_completed

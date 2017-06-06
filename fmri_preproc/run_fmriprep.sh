set -e
ignore_list=ignore_list.txt
for path in /oak/stanford/groups/russpold/data/uh2/sub*
do
    echo "*******************************************"
    sid=${path:(-4)}
    if grep -Fxq "$sid" $ignore_list; then
        echo $sid is being ignored
    else
        if [[ -d /scratch/PI/russpold/work/ieisenbe/uh2/fmriprep/fmriprep/sub-${sid}/ses-1 && 
            -d /scratch/PI/russpold/work/ieisenbe/uh2/fmriprep/fmriprep/sub-${sid}/ses-2 ]]; then
            echo fmriprep already run on $sid
        else
            echo running fmriprep $sid
            sed "s/{sid}/$sid/g" fmriprep.batch | sbatch -p russpold
        fi
    fi
done



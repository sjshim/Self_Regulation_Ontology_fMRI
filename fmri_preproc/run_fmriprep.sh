set -e
ignore_list=ignore_list.txt
subjects_completed=0
subjects_run=0
for path in /oak/stanford/groups/russpold/data/uh2/sub*
do
    sid=${path:(-4)}
    echo "*******************${sid}************************"
    if grep -Fxq "$sid" $ignore_list; then
        echo $sid is being ignored
    else
        check_fmriprep=0
        for session in 1 2 3
        do
            # if a session exists in data, check that the directory exists in fmriprep
            if [[  -d /oak/stanford/groups/russpold/data/uh2/sub-${sid}/ses-${session} ]]; then
                num_epi=$(ls /oak/stanford/groups/russpold/data/uh2/sub-${sid}/ses-${session}/func/*task*bold.nii.gz | wc -l)
                if [[ -d /scratch/PI/russpold/work/ieisenbe/uh2/fmriprep/fmriprep/sub-${sid}/ses-${session} ]]; then
                    num_preproc=$(ls /scratch/PI/russpold/work/ieisenbe/uh2/fmriprep/fmriprep/sub-${sid}/ses-${session}/func/*MNI*preproc.nii.gz | wc -l)
                    echo fmriprep session ${session} run
                    if [ $num_epi -ne $num_preproc ]; then
                        echo Number of task scans \($num_epi\) does not equal number of preprocessed scans \($num_preproc\)
                        check_fmriprep+=1
                    fi
                else
                    check_fmriprep+=1
                fi
            fi
        done
        if [[ $check_fmriprep>0 ]]; then
            echo "**Running fmriprep on $sid**"
            sed "s/{sid}/$sid/g" fmriprep.batch | sbatch -p russpold
            (( subjects_run+=1 ))
        else
            (( subjects_completed+=1 ))
        fi

    fi
done
echo Subjects running: $subjects_run, subjects completed: $subjects_completed


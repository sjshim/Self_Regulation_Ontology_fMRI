set -e

fmriprep_path=`sed '4q;d' singularity_config.txt`
data_path=`sed '6q;d' singularity_config.txt`
out_path=`sed '8q;d' singularity_config.txt`

ignore_list=ignore_list.txt
subjects_completed=0
subjects_run=0
for path in ${data_path}/sub*
do
    sid=${path:(-4)}
    echo "*******************${sid}************************"
    if grep -Fxq "$sid" $ignore_list; then
        echo $sid is being ignored
    else
        check_fmriprep=0
        # if a session exists in data, check that the directory exists in fmriprep
        if [[  -d ${data_path}/sub-${sid}/ses-1 ]]; then
            if [[ -f ${out_path}/fmriprep/freesurfer/sub-${sid}/mri/aseg.mgz ]]; then
                echo fmriprep-anat session ${session} run
            else
                check_fmriprep=1
            fi
        fi
        # if no T1 do not run
        num_T1=$(ls ${data_path}/sub-${sid}/ses-*/anat/*T1* | wc -l)
        if [ $num_T1 -eq 0 ]; then
            echo no T1 found for ${sid}! Cannot run fmriprep
            check_fmriprep=0
        else
            if [[ $check_fmriprep>0 ]]; then
                echo "**Running fmriprep-anat on $sid**"
                echo singularity run ${fmriprep_path} ${data_path} ${out_path}/fmriprep participant --participant_label ${sid} -w $SCRATCH --fs-license-file ~/docs/fs-license.txt --output-space template T1w --mem_mb 40000 --nthreads 10 --anat-only >> preproc_task_list.sh
                (( subjects_run+=1 ))
            else
                (( subjects_completed+=1 ))
            fi
        fi

    fi
done
echo Subjects running: $subjects_run, subjects completed: $subjects_completed

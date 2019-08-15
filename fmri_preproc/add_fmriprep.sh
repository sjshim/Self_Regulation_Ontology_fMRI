set -e

fmriprep_path=`sed '4q;d' singularity_config.txt`
data_path=`sed '6q;d' singularity_config.txt`
out_path=`sed '8q;d' singularity_config.txt`

ignore_list=ignore_list.txt
subjects_completed=0
no_freesurfer=0
subjects_run=0
for path in ${data_path}/sub*
do
    sid=${path:(-4)}
    echo "*******************${sid}************************"
    if grep -Fxq "$sid" $ignore_list; then
        echo $sid is being ignored
    else
        check_fmriprep=0
        if [[ -f ${out_path}/fmriprep/freesurfer/sub-${sid}/mri/aseg.mgz ]]; then
            for session in 1 2 3 4
            do
                # if a session exists in data, check that the directory exists in fmriprep
                if [[  -d ${data_path}/sub-${sid}/ses-${session} ]]; then
                    num_epi=$(ls ${data_path}/sub-${sid}/ses-${session}/func/*task*bold.nii.gz | wc -l)
                    if [[ -d ${out_path}/fmriprep/fmriprep/sub-${sid}/ses-${session} ]]; then
                        num_preproc=$(ls ${out_path}/fmriprep/fmriprep/sub-${sid}/ses-${session}/func/*MNI*preproc_bold.nii.gz | wc -l)
                        echo fmriprep session ${session} run
                        if [ $num_epi -ne $num_preproc ]; then
                            echo Number of task scans \($num_epi\) does not equal number of preprocessed scans \($num_preproc\)
                            check_fmriprep=1
                        fi
                    else
                        check_fmriprep=1
                    fi
                fi
            done
        else
            check_fmriprep=-1
            echo Freesurfer has not successfully been run
        fi
        # if no T1 do not run
        num_T1=$(ls ${data_path}/sub-${sid}/ses-*/anat/*T1* | wc -l)
        if [ $num_T1 -eq 0 ]; then
            echo no T1 found for ${sid}! Cannot run fmriprep
            check_fmriprep=0
        else
            if [[ $check_fmriprep -eq 1 ]]; then
                echo "**Running fmriprep on $sid**"
                echo singularity run ${fmriprep_path} ${data_path} ${out_path}/fmriprep participant --participant_label ${sid} -w $LOCAL_SCRATCH --fs-license-file ~/docs/fs-license.txt --output-space template T1w fsaverage --template-resampling-grid 2mm --mem_mb 40000 --nthreads 10 --omp-nthread 8 >> preproc_task_list.sh
                (( subjects_run+=1 ))
            elif [[ $check_fmriprep -eq -1 ]]; then
                (( no_freesurfer+=1 ))
            else
                (( subjects_completed+=1 ))
            fi
        fi

    fi
done
echo Subjects running: $subjects_run, subjects w/o freesurfer: $no_freesurfer, subjects completed: $subjects_completed

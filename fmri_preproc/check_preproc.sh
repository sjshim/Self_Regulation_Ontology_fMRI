set -e
ignore_list=ignore_list.txt
mriqc_subjects_run=0
mriqc_subjects_completed=0
fmriprep_subjects_run=0
fmriprep_subjects_completed=0

for path in /oak/stanford/groups/russpold/data/uh2/sub*
do
    echo "*******************************************"
    sid=${path:(-4)}
    echo $sid
    echo "*******************************************"
    if grep -Fxq "$sid" $ignore_list; then
        echo $sid is being ignored
    else
        check_mriqc=0
        check_fmriprep=0

        # check mriqc
        for session in 1 2 3
        do
        # if a session exists in data, check that some files exist in the corresponding mriqc directory
        if [ -d /oak/stanford/groups/russpold/data/uh2/sub-${sid}/ses-${session} ]; then
            # number of epi scans found in session folder
            num_epi=$(ls /oak/stanford/groups/russpold/data/uh2/sub-${sid}/ses-${session}/func/*task*bold.nii.gz | wc -l)
            # check mriqc
            mriqc_files=( $(find /scratch/PI/russpold/work/ieisenbe/uh2/mriqc/reports/ -name "*${sid}*ses-${session}*run-*") )
            if [[ ${#mriqc_files[@]} -ne 0 ]]; then
                echo mriqc session ${session} run
                if [[ ${#mriqc_files[@]} -ne $num_epi ]]; then
                    echo Number of task scans \($num_epi\) does not equal number of mriqc reports \(${#files[@]}\)
                fi
            else
                check_mriqc+=1
            fi
        fi
        done
        if [[ $check_mriqc>0 ]]; then
            echo "** MRIQC needs to be run on $sid **" 
            (( mriqc_subjects_run+=1 ))
        else
            echo mriqc run on $sid
            (( mriqc_subjects_completed+=1 ))
        fi

        echo ""

        # now check fmriprep
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
        # if no T1 do not run
        num_T1=$(ls /oak/stanford/groups/russpold/data/uh2/sub-${sid}/ses-*/anat/*T1* | wc -l)
        if [ $num_T1 -eq 0 ]; then
            echo no T1 found for ${sid}! Cannot run fmriprep
            check_fmriprep=0
        else
            if [[ $check_fmriprep>0 ]]; then
                echo "** fmriprep needs to be run $sid **"
                (( fmriprep_subjects_run+=1 ))   
            else
                echo fmriprep run on $sid
                (( fmriprep_subjects_completed+=1 ))
            fi
        fi
    fi
done

echo MRIQC: Subjects running: $mriqc_subjects_run, subjects completed: $mriqc_subjects_completed
echo ""
echo fmriprep: Subjects running: $fmriprep_subjects_run, subjects completed: $fmriprep_subjects_completed

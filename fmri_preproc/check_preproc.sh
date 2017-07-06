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
    sid=${path:(-4)}
    if grep -Fxq "$sid" $ignore_list; then
        echo $sid is being ignored
    else
        check_mriqc=0
        check_fmriprep=0

        # check mriqc
        # if a session exists in data, check that some files exist in the corresponding mriqc directory
        if [[  -d /oak/stanford/groups/russpold/data/uh2/sub-${sid}/ses-1 ]]; then
            # number of epi scans found in session folder
            num_epi=$(ls /oak/stanford/groups/russpold/data/uh2/sub-${sid}/ses-1/func/*task*bold.nii.gz | wc -l)
            # check mriqc
            mriqc_files=( $(find /scratch/PI/russpold/work/ieisenbe/uh2/mriqc/reports/ -name "*${sid}*ses-1*run-*") )
            if [[ ${#mriqc_files[@]} -ne 0 ]]; then
                echo mriqc session 1 run
                if [[ ${#mriqc_files[@]} -ne $num_epi ]]; then
                    echo Number of task scans \($num_epi\) does not equal number of mriqc reports \(${#files[@]}\)
                fi
            else
                check_mriqc+=1
            fi
        fi
        # session 2
        if [[  -d /oak/stanford/groups/russpold/data/uh2/sub-${sid}/ses-2 ]]; then
            num_epi=$(ls /oak/stanford/groups/russpold/data/uh2/sub-${sid}/ses-2/func/*task*bold.nii.gz | wc -l)
            # check mriqc
            mriqc_files=( $(find /scratch/PI/russpold/work/ieisenbe/uh2/mriqc/reports/ -name "*${sid}*ses-2*run-*") )
            if [[ ${#mriqc_files[@]} -ne 0 ]]; then
                echo mriqc session 2 run
                if [[ ${#mriqc_files[@]} -ne $num_epi ]]; then
                    echo Number of task scans \($num_epi\) does not equal number of mriqc reports \(${#files[@]}\)
                fi
            else
                check_mriqc+=1
            fi
        fi
        # session 3 (for a few subjects)
        if [[  -d /oak/stanford/groups/russpold/data/uh2/sub-${sid}/ses-3 ]]; then
            num_epi=$(ls /oak/stanford/groups/russpold/data/uh2/sub-${sid}/ses-3/func/*task*bold.nii.gz | wc -l)
            # check mriqc
            mriqc_files=( $(find /scratch/PI/russpold/work/ieisenbe/uh2/mriqc/reports/ -name "*${sid}*ses-3*run-*") )
            if [[ ${#mriqc_files[@]} -ne 0 ]]; then
                echo mriqc session 3 run
                if [[ ${#mriqc_files[@]} -ne $num_epi ]]; then
                    echo Number of task scans \($num_epi\) does not equal number of mriqc reports \(${#files[@]}\)
                fi
            else
                check_mriqc+=1
            fi
        fi

        if [[ $check_mriqc>0 ]]; then
            echo "** MRIQC needs to be run on $sid **" 
            (( mriqc_subjects_run+=1 ))
        else
            echo mriqc run on $sid
            (( mriqc_subjects_completed+=1 ))
        fi

        echo ""

        # now check fmriprep
        # if a session exists in data, check that the directory exists in fmriprep
        if [[  -d /oak/stanford/groups/russpold/data/uh2/sub-${sid}/ses-1 ]]; then
            num_epi=$(ls /oak/stanford/groups/russpold/data/uh2/sub-${sid}/ses-1/func/*task*bold.nii.gz | wc -l)
            if [[ -d /scratch/PI/russpold/work/ieisenbe/uh2/fmriprep/fmriprep/sub-${sid}/ses-1 ]]; then
                num_preproc=$(ls /scratch/PI/russpold/work/ieisenbe/uh2/fmriprep/fmriprep/sub-${sid}/ses-1/func/*MNI*preproc.nii.gz | wc -l)
                echo fmriprep session 1 run
                if [ $num_epi -ne $num_preproc ]; then
                    echo Number of task scans \($num_epi\) does not equal number of preprocessed scans \($num_preproc\)
                fi
            else
                check_fmriprep+=1
            fi
        fi
        # session 2
        if [[  -d /oak/stanford/groups/russpold/data/uh2/sub-${sid}/ses-2 ]]; then
            num_epi=$(ls /oak/stanford/groups/russpold/data/uh2/sub-${sid}/ses-2/func/*task*bold.nii.gz | wc -l)
            if [[ -d /scratch/PI/russpold/work/ieisenbe/uh2/fmriprep/fmriprep/sub-${sid}/ses-2 ]]; then
                num_preproc=$(ls /scratch/PI/russpold/work/ieisenbe/uh2/fmriprep/fmriprep/sub-${sid}/ses-2/func/*MNI*preproc.nii.gz | wc -l)
                echo fmriprep session 2 run
                if [ $num_epi -ne $num_preproc ]; then
                    echo Number of task scans \($num_epi\) does not equal number of preprocessed scans \($num_preproc\)
                fi
            else
                check_fmriprep+=1
            fi
        fi
        # session 3 (for a few subjects)
        if [[  -d /oak/stanford/groups/russpold/data/uh2/sub-${sid}/ses-3 ]]; then
            num_epi=$(ls /oak/stanford/groups/russpold/data/uh2/sub-${sid}/ses-3/func/*task*bold.nii.gz | wc -l)
            if [[ -d /scratch/PI/russpold/work/ieisenbe/uh2/fmriprep/fmriprep/sub-${sid}/ses-3 ]]; then
                num_preproc=$(ls /scratch/PI/russpold/work/ieisenbe/uh2/fmriprep/fmriprep/sub-${sid}/ses-3/func/*MNI*preproc.nii.gz | wc -l)
                echo fmriprep session 3 run
                if [ $num_epi -ne $num_preproc ]; then
                    echo Number of task scans \($num_epi\) does not equal number of preprocessed scans \($num_preproc\)
                fi
            else
                check_fmriprep+=1
            fi
        fi
        if [[ $check_fmriprep>0 ]]; then
            echo "** fmriprep needs to be run $sid **"
            (( fmriprep_subjects_run+=1 ))   
        else
            echo fmriprep run on $sid
            (( fmriprep_subjects_completed+=1 ))
        fi
    fi
done

echo MRIQC: Subjects running: $mriqc_subjects_run, subjects completed: $mriqc_subjects_completed
echo ""
echo fmriprep: Subjects running: $fmriprep_subjects_run, subjects completed: $fmriprep_subjects_completed
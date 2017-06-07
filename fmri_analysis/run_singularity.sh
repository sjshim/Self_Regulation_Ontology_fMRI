#singularity run -B `pwd`/output:/scratch -B `pwd`/Data:/Data singularity_images/nipype_image-2017-06-06-d2bbd7987fd2.img --participant_label $1 

singularity run -B /home/ieisenbe/Self_Regulation_Ontology/fmri_analysis/output:/scratch -B /scratch/PI/russpold/work/ieisenbe/uh2/fmriprep/fmriprep/:/Data singularity_images/nipype_image-2017-06-06-d2bbd7987fd2.img --participant_label s358

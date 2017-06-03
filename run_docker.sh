docker run -ti --rm -p 8888:8888 -v `pwd`/fmri_analysis/Data:/Data:ro  nipype_image --participant_label $1

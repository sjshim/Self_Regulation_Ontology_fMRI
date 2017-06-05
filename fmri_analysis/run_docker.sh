docker run -ti --rm -p 8888:8888 -v `pwd`/output:/scratch -v `pwd`/Data:/Data:ro  nipype_image --participant_label $1 --tasks stroop


#docker run -ti --rm -p 8888:8888 -v `pwd`/Docker:/home/scripts -v `pwd`/Data:/Data:ro  --entrypoint=/bin/bash nipype_image --participant_label $1 --tasks stroop

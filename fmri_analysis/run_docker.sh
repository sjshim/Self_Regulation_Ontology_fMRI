docker run -ti --rm -p 8888:8888 -v `pwd`/output:/scratch -v `pwd`/Data:/Data:ro  nipype_image --participant_label $1 


# command to ignore entrypoint and access docker container
#docker run -ti --rm -p 8888:8888 -v `pwd`/Docker:/home/scripts -v `pwd`/output:/scratch -v `pwd`/Data:/Data:ro  --entrypoint=/bin/bash nipype_image 

# command to build Docker image from the "Docker" folder
# docker build -t nipype_image .

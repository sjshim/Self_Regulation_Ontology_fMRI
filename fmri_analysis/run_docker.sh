# example to run 1st level analysis
docker run -ti --rm -p 8888:8888 -v `pwd`/output:/output -v /mnt/Data/:/Data:ro  nipype_image  /output --nipype_image  --tasks stroop

# example to run 2nd level analysis
docker run -ti --rm -p 8888:8888 -v `pwd`/output/Plots:/output -v /mnt/Sherlock_Scratch/output/1stLevel:/Data:ro  nipype_image /output --script group_plots.py --tasks stroop  


# command to ignore entrypoint and access docker container
# 1st level
#docker run -ti --rm -p 8888:8888 -v `pwd`/Docker/scripts:/scripts -v `pwd`/output:/output -v  /mnt:/Data:ro  --entrypoint=/bin/bash nipype_image 

# 2nd level
#docker run -ti --rm -p 8888:8888 -v `pwd`/Docker/scripts:/scripts -v `pwd`/output:/output -v  /mnt/Sherlock_Scratch/output/1stLevel:/Data:ro  --entrypoint=/bin/bash nipype_image 

# command to build Docker image from the "Docker" folder
# docker build -t nipype_image .
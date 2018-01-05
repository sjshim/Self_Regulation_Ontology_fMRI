# example to run 1st level analysis
docker run -ti --rm -v ~/Experiments/expfactory/Self_Regulation_Ontology_fMRI/fmri_analysis:/home -v $HOME/temp:/output -v /mnt/OAK/:/Data sro_fmri python scripts/task_analysis.py /output /Data --participant s358 --tasks stopSignal 

# example to run 2nd level analysis
docker run -ti --rm -p 8888:8888 -v `pwd`/Docker/scripts:/scripts -v `pwd`/output/:/output -v /mnt/Sherlock_Scratch/output/1stLevel:/scratch:ro  -v /mnt/Data/:/Data:ro nipype_image /output --script group_analysis.py --tasks stroop --mask_dir /Data --data_dir /scratch --tasks stroop

# example to run 2nd level plotting
docker run -ti --rm -p 8888:8888 -v `pwd`/Docker/scripts:/scripts -v `pwd`/output/:/output -v /mnt/Sherlock_Scratch/output/2ndLevel/custom_modeling:/scratch:ro  -v /mnt/Data/:/Data:ro nipype_image /output --script group_plots.py --tasks stroop --data_dir /scratch --tasks stroop

# command to ignore entrypoint and access docker container
# 1st level
#docker run -ti --rm -p 8888:8888 -v `pwd`/Docker/scripts:/scripts -v `pwd`/output:/output -v  /mnt:/Data:ro  --entrypoint=/bin/bash nipype_image 

# 2nd level
# docker run -ti --rm -p 8888:8888 -v `pwd`/output/:/output -v /mnt/Sherlock_Scratch/output/1stLevel:/scratch:ro  -v /mnt/Data/:/Data:ro -v `pwd`/Docker/scripts:/scripts --entrypoint=/bin/bash nipype_image  


# command to build Docker image from the "Docker" folder
# docker build --rm -t sro_fmri .
sshfs sherlock:/scratch/PI/russpold/work/ieisenbe/uh2/fmriprep/fmriprep /mnt/Sherlock_Scratch
cp -r /mnt/Sherlock_Scratch/*${1} .
sudo umount /mnt/Sherlock_Scratch
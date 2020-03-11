# Self_Regulation_Ontology - fMRI analysis

### Setting up the docker image
This is the recommended way to use the repo

Run: 
```
docker build --rm -t fmri_env . 
```
That's it!

In docker\_files/run_docker.txt you can find example docker commands.
If you start the notebook version, 
you can access it at the following url:
http://127.0.0.1:8888/lab?

### Setting up python environment

conda create -n SRO python=3.5.3
source activate SRO
pip install -r requirements.txt
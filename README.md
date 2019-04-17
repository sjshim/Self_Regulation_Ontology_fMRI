# Self_Regulation_Ontology - fMRI analysis

### Setting up the docker image
This is the recommended way to use the repo
Run: docker build --rm -t fmri_env .
That's it!

In docker_files/run_docker.txt you can find example docker commands.
If you start the notebook version, you can access it at the following url:
http://127.0.0.1:8888/lab?

### Setting up python environment

Use the environment.yml file with anaconda: conda install -f environment.yml

After doing that, you must install expanalysis in the same environment.
- Clone expanalysis from: https://github.com/IanEisenberg/expfactory-analysis
- Enter expanalysis and enter "pip install -e ."

Finally you must install the selfregulation python: python setup.py install

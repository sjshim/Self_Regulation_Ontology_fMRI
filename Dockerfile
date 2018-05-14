# Use Ubuntu 16.04 LTS
FROM jupyter/base-notebook
USER root

# Pre-cache neurodebian key
COPY docker_files/neurodebian.gpg /root/.neurodebian.gpg

# Prepare environment
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    curl \
                    bzip2 \
                    ca-certificates \
                    xvfb \
                    cython3 \
                    build-essential \
                    autoconf \
                    libtool \
                    pkg-config && \
    curl -sSL http://neuro.debian.net/lists/xenial.us-ca.full >> /etc/apt/sources.list.d/neurodebian.sources.list && \
    apt-key add /root/.neurodebian.gpg && \
    (apt-key adv --refresh-keys --keyserver hkp://ha.pool.sks-keyservers.net 0xA5D32F012649A5A9 || true) && \
    apt-get update

# Installing Neurodebian packages (AFNI)
RUN apt-get update && \
    apt-get install -y --no-install-recommends afni=16.2.07~dfsg.1-5~nd16.04+1

ENV AFNI_MODELPATH=/usr/lib/afni/models \
    AFNI_IMSAVE_WARNINGS=NO \
    AFNI_TTATLAS_DATASET=/usr/share/afni/atlases \
    AFNI_PLUGINPATH=/usr/lib/afni/plugins
ENV PATH=/usr/lib/afni/bin:$PATH

# Installing ANTs 2.2.0 (NeuroDocker build)
ENV ANTSPATH=/usr/lib/ants
RUN mkdir -p $ANTSPATH && \
    curl -sSL "https://dl.dropbox.com/s/2f4sui1z6lcgyek/ANTs-Linux-centos5_x86_64-v2.2.0-0740f91.tar.gz" \
    | tar -xzC $ANTSPATH --strip-components 1
ENV PATH=$ANTSPATH:$PATH

# Installing and setting up c3d
RUN mkdir -p /opt/c3d && \
    curl -sSL "http://downloads.sourceforge.net/project/c3d/c3d/1.0.0/c3d-1.0.0-Linux-x86_64.tar.gz" \
    | tar -xzC /opt/c3d --strip-components 1

ENV C3DPATH /opt/c3d/
ENV PATH $C3DPATH/bin:$PATH

# Installing WEBP tools
RUN curl -sSLO "http://downloads.webmproject.org/releases/webp/libwebp-0.5.2-linux-x86-64.tar.gz" && \
  tar -xf libwebp-0.5.2-linux-x86-64.tar.gz && cd libwebp-0.5.2-linux-x86-64/bin && \
  mv cwebp /usr/local/bin/ && rm -rf libwebp-0.5.2-linux-x86-64

# Installing SVGO
RUN curl -sL https://deb.nodesource.com/setup_7.x | bash -
RUN apt-get install -y nodejs
RUN npm install -g svgo

# Installing and setting up ICA_AROMA
RUN mkdir -p /opt/ICA-AROMA && \
  curl -sSL "https://github.com/rhr-pruim/ICA-AROMA/archive/v0.4.1-beta.tar.gz" \
  | tar -xzC /opt/ICA-AROMA --strip-components 1 && \
  chmod +x /opt/ICA-AROMA/ICA_AROMA.py

ENV PATH=/opt/ICA-AROMA:$PATH

# Installing precomputed python packages
RUN conda install -y mkl=2017.0.1 mkl-service &&  \
    conda install -y numpy=1.12.0 \
                     scipy=0.18.1 \
                     scikit-learn=0.18.1 \
                     matplotlib=2.0.0 \
                     pandas=0.19.2 \
                     ipython=5.1.0 \
                     seaborn=0.7.1 \
                     libxml2=2.9.4 \
                     libxslt=1.1.29\
                     traits=4.6.0 &&  \
    chmod +x $CONDA_DIR/* && \
    conda clean --all -y && \
    python -c "from matplotlib import font_manager" && \
    sed -i 's/\(backend *: \).*$/\1Agg/g' $( python -c "import matplotlib; print(matplotlib.matplotlib_fname())" )

# Precaching fonts
RUN python -c "from matplotlib import font_manager"

# Installing and configuring FSL
COPY docker_files/fslinstaller.py /home/fslinstaller.py
RUN python2 /home/fslinstaller.py --quiet --dest=/usr/share/fsl -E

ENV FSLDIR=/usr/share/fsl
RUN . ${FSLDIR}/etc/fslconf/fsl.sh
ENV PATH=${FSLDIR}/bin:${PATH}

# Installing Ubuntu packages and cleaning up
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    git=1:2.7.4-0ubuntu1 \
                    graphviz=2.38.0-12ubuntu2 && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Unless otherwise specified each process should only use one thread - nipype
# will handle parallelization
ENV MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

# Installing dev requirements (packages that are not in pypi)
WORKDIR /home
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt && \
    rm -rf ~/.cache/pip

# Set up data and script directories
Run mkdir /scripts
WORKDIR /scripts



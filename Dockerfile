FROM nvidia/cuda:9.0-base-ubuntu16.04

LABEL maintainer="Seyoung Park <seyoung.arts.park@protonmail.com>"

# This Dockerfile is forked from Tensorflow Dockerfile

# Pick up some PyTorch gpu dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-9-0 \
        cuda-cublas-9-0 \
        cuda-cufft-9-0 \
        cuda-curand-9-0 \
        cuda-cusolver-9-0 \
        cuda-cusparse-9-0 \
        curl \
        libcudnn7=7.1.4.18-1+cuda9.0 \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python \
        python-dev \
        rsync \
        software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Install miniconda
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget && \
    MINICONDA="Miniconda3-latest-Linux-x86_64.sh" && \
    wget --quiet https://repo.continuum.io/miniconda/$MINICONDA && \
    bash $MINICONDA -b -p /miniconda && \
    rm -f $MINICONDA
ENV PATH /miniconda/bin:$PATH

# Install PyTorch
RUN conda update -n base conda && \ 
    conda install pytorch torchvision cuda90 -c pytorch

# Install PerceptualSimilarity dependencies
RUN conda install numpy scipy jupyter matplotlib && \
    conda install -c conda-forge scikit-image && \
    apt-get install -y python-qt4 && \
    pip install opencv-python

# For CUDA profiling, TensorFlow requires CUPTI. Maybe PyTorch needs this too.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# IPython
EXPOSE 8888

WORKDIR "/notebooks"


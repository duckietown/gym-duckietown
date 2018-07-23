FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04

RUN apt-get update -y && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         ca-certificates \
         libjpeg-dev \
         libpng-dev \
         xvfb \
         freeglut3-dev && \
     rm -rf /var/lib/apt/lists/*

ADD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh miniconda.sh
RUN sh miniconda.sh -b -p /opt/conda && rm miniconda.sh
ENV PATH /opt/conda/bin:$PATH

RUN conda install numpy pyyaml scipy ipython mkl mkl-include && \
    conda install -c pytorch magma-cuda91 torchvision && \
    conda clean -ya

# This must be done before pip so that requirements.txt is available
WORKDIR /gym-duckietown
COPY . .

RUN TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    pip install -v -e .

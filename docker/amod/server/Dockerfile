FROM nvidia/cuda:9.1-runtime-ubuntu16.04

RUN apt-get update -y && apt-get install -y --no-install-recommends \
         git \
         xvfb \
         bzip2 \
         freeglut3-dev && \
     rm -rf /var/lib/apt/lists/*

ADD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh miniconda.sh
RUN sh miniconda.sh -b -p /opt/conda && rm miniconda.sh
ENV PATH /opt/conda/bin:$PATH

RUN conda install numpy pyyaml scipy ipython mkl mkl-include pyzmq && conda clean -ya

WORKDIR /workspace
RUN chmod -R a+w /workspace

ADD . gym-duckietown
RUN pip install -e gym-duckietown

RUN pip install -e git+https://github.com/duckietown/duckietown-slimremote.git#egg=duckietown-slimremote

COPY docker/amod/server/launch-gym-server-with-xvfb.sh /usr/bin/launch-gym-server-with-xvfb

EXPOSE 5558 8902

CMD launch-gym-server-with-xvfb

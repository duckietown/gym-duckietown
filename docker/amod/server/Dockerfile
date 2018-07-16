FROM ubuntu:16.04

# TODO: double-check that we really need all of these for just the server
RUN apt-get update -y && apt-get install -y --no-install-recommends \
         git \
         xvfb \
         bzip2 \
         ca-certificates \
         freeglut3-dev && \
     rm -rf /var/lib/apt/lists/*

ADD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh miniconda.sh
RUN sh miniconda.sh -b -p /opt/conda && rm miniconda.sh
ENV PATH /opt/conda/bin:$PATH

RUN conda install numpy pyyaml scipy pyzmq && conda clean -ya

WORKDIR /workspace
RUN chmod -R a+w /workspace
RUN git clone http://github.com/duckietown/duckietown-slimremote && cd duckietown-slimremote && pip install -e .
RUN git clone http://github.com/duckietown/gym-duckietown && cd gym-duckietown && pip install -e .

EXPOSE 5558 8902
ENTRYPOINT ["duckietown-start-gym"]

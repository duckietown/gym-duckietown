FROM nvidia/cuda:9.1-runtime-ubuntu16.04

RUN apt-get update -y && apt-get install -y --no-install-recommends \
         git \
         xvfb \
         bzip2 \
         python-pip \
         python-setuptools\
         freeglut3-dev \
          python-subprocess32 python-matplotlib python-yaml python-opencv
#         && \
#     rm -rf /var/lib/apt/lists/*

WORKDIR /workspace



COPY docker/amod/server-python2/requirements.txt /requirements.txt
RUN pip install -r /requirements.txt


RUN pip install -e git+https://github.com/duckietown/duckietown-slimremote.git#egg=duckietown-slimremote


EXPOSE 5558 8902



ADD . gym-duckietown

RUN cd gym-duckietown && python setup.py develop --no-deps

COPY docker/amod/server-python2/launch-gym-server-with-xvfb.sh /usr/bin/launch-gym-server-with-xvfb
COPY docker/amod/server-python2/launch-xvfb /usr/bin/launch-xvfb

CMD launch-gym-server-with-xvfb

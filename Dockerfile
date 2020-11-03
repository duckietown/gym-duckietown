ARG AIDO_REGISTRY

FROM ${AIDO_REGISTRY}/duckietown/aido-base-python3:daffy-amd64


RUN apt-get update -y && \
    apt-get install -y --no-install-recommends xvfb freeglut3-dev libglib2.0-dev libgtk2.0-dev git

WORKDIR /gym-duckietown

ARG PIP_INDEX_URL
ENV PIP_INDEX_URL=${PIP_INDEX_URL}

RUN pip install -U "pip>=20.2"
## first install the ones that do not change
COPY requirements.pin.txt .
RUN pip install --use-feature=2020-resolver -r requirements.pin.txt
#
COPY requirements.* ./
RUN cat requirements.* > .requirements.txt
RUN cat .requirements.txt

RUN pip install --use-feature=2020-resolver -r .requirements.txt

COPY . .

RUN pip install -v  --no-deps -e .

#   pip install -v -e .

RUN pipdeptree

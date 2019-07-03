FROM pytorch/pytorch

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    xvfb \
    libglib2.0-0 \
    libxrender-dev \
    freeglut3-dev && \
    rm -rf /var/lib/apt/lists/*
 
RUN conda install numpy pyyaml scipy ipython mkl mkl-include && conda clean -ya

RUN pip install --upgrade setuptools dataclasses gym zmq geometry pygeometry opencv-python pillow

WORKDIR /gym-duckietown

COPY . .

# Export your display id
ENV DISPLAY=:1

CMD learning/start-xvbf.sh && python learning/reinforcement/pytorch/a3c/train_reinforcement_discrete.py

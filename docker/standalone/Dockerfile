FROM pytorch/pytorch

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    xvfb \
    libglib2.0-0 \
    libxrender-dev \
    freeglut3-dev && \
    rm -rf /var/lib/apt/lists/*
 
RUN conda install numpy pyyaml scipy ipython mkl mkl-include && conda clean -ya

RUN pip install --upgrade setuptools dataclasses geometry pygeometry

WORKDIR /gym-duckietown

COPY . .

RUN TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    pip install -v -e .

# Export your display id
ENV DISPLAY=:1

CMD learning/start-xvbf.sh && python learning/reinforcement/pytorch/a3c/train_reinforcement_discrete.py

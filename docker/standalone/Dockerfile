#FROM pytorch/pytorch
FROM nvidia/opengl:1.2-glvnd-devel
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    xvfb \
    freeglut3-dev && \
    rm -rf /var/lib/apt/lists/*

#RUN conda install numpy pyyaml scipy ipython mkl mkl-include && conda clean -ya

WORKDIR /gym-duckietown

COPY . .

RUN TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    pip install -v -e .

RUN pip install pyglet==1.5.15
RUN python3 -c "from gym_duckietown import *"

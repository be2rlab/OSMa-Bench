FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

SHELL [ "/bin/bash", "-c" ]
ENV DEBIAN_FRONTEND noninteractive
ENV USER=docker_user

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        sudo git-all wget nano unzip ffmpeg libsm6 libxext6 \
        build-essential python3-dev libopenblas-dev libglew-dev && \
    rm -rf /var/lib/apt/lists/*

ARG USE_CUDA=0
ARG TORCH_ARCH="5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
ENV AM_I_DOCKER True
ENV BUILD_WITH_CUDA "${USE_CUDA}"
ENV TORCH_CUDA_ARCH_LIST "${TORCH_ARCH}"

ENV CONDA_DIR /opt/conda
ENV PATH=${CONDA_DIR}/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -u -b -p /opt/conda

RUN conda create -n openscene_env python=3.10
RUN conda install -y -n openscene_env pytorch==2.1.2 torchvision==0.16.2 pytorch-cuda=12.1 -c pytorch -c nvidia
RUN conda run -n openscene_env python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
RUN conda install -y -n openscene_env openblas-devel -c anaconda
RUN conda run -n openscene_env python -m pip install ninja==1.10.2.3

WORKDIR /home/${USER}
RUN git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
RUN cd MinkowskiEngine && \
    git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228 && \
    sed -i '31i #include <thrust/execution_policy.h>' ./src/convolution_kernel.cuh && \
    sed -i '39i #include <thrust/unique.h>\n#include <thrust/remove.h>' ./src/coordinate_map_gpu.cu && \
    sed -i '38i #include <thrust/execution_policy.h>\n#include <thrust/reduce.h>\n#include <thrust/sort.h>' ./src/spmm.cu &&\
    sed -i '38i #include <thrust/execution_policy.h>' ./src/3rdparty/concurrent_unordered_map.cuh && \
    conda run -n openscene_env python setup.py install --force_cuda --blas=openblas

RUN conda run -n openscene_env python -m pip install scipy open3d ftfy tensorflow tensorboardx
RUN conda run -n openscene_env python -m pip install tqdm imageio plyfile opencv-python 
RUN conda run -n openscene_env python -m pip install sharedarray
RUN conda run -n openscene_env python -m pip install numpy==1.26.4

RUN conda run -n openscene_env python -m pip install git+https://github.com/openai/CLIP.git
RUN conda run -n openscene_env python -m pip install open_clip_torch

RUN conda install -y -n openscene_env https://anaconda.org/pytorch3d/pytorch3d/0.7.5/download/linux-64/pytorch3d-0.7.5-py310_cu121_pyt210.tar.bz2

ARG UID=1000
ARG GID=1000
RUN useradd -m ${USER} --uid=${UID} \
    && usermod -s /bin/bash ${USER} \
    && usermod -a -G sudo ${USER} \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
WORKDIR /home/${USER}/OpenScene
RUN chown -R $USER /home/${USER}
USER ${USER}

RUN echo 'source activate openscene_env' >> /home/${USER}/.bashrc
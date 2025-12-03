FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel AS base

SHELL [ "/bin/bash", "-c" ]
ENV DEBIAN_FRONTEND noninteractive
ENV USER=docker_user

RUN apt-get update -y && \
    apt install -y git-all wget nano unzip ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update -y && apt-get install -y python3-tk

ENV CONDA_DIR /opt/conda
ENV PATH=${CONDA_DIR}/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -u -b -p /opt/conda

RUN conda create -n cg_env python=3.10

ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
ENV PYTHONPATH "${PYTHONPATH}:/opt/src"
ENV PATH="${PATH}:/opt/hpcx/ompi/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/hpcx/ompi/lib"
ENV PATH="${PATH}:/opt/hpcx/ucx/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/hpcx/ucx/lib"

RUN conda install -y -n cg_env -c pytorch faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl
RUN conda install -y -n cg_env pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
RUN conda install -y -n cg_env https://anaconda.org/pytorch3d/pytorch3d/0.7.5/download/linux-64/pytorch3d-0.7.5-py310_cu121_pyt210.tar.bz2

RUN conda run -n cg_env python -m pip install tyro open_clip_torch wandb h5py openai hydra-core

WORKDIR /home/${USER}
RUN conda run -n cg_env python -m pip install chamferdist

RUN git clone https://github.com/gradslam/gradslam.git -b conceptfusion
RUN conda run -n cg_env python -m pip install -e gradslam/

ARG USE_CUDA=0
ARG TORCH_ARCH="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
ENV AM_I_DOCKER True
ENV BUILD_WITH_CUDA "${USE_CUDA}"
ENV TORCH_CUDA_ARCH_LIST "${TORCH_ARCH}"
ENV CUDA_HOME /usr/local/cuda-12.1/

RUN git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git

RUN apt-get update -y && apt-get install --no-install-recommends wget ffmpeg=7:* \
    libsm6=2:* libxext6=2:* -y \
    && apt-get clean && apt-get autoremove && rm -rf /var/lib/apt/lists/*

RUN conda run -n cg_env python -m pip install --no-cache-dir -e Grounded-Segment-Anything/segment_anything
RUN conda run -n cg_env python -m pip install --no-cache-dir wheel
RUN conda run -n cg_env python -m pip install --no-cache-dir --no-build-isolation -e Grounded-Segment-Anything/GroundingDINO

RUN conda run -n cg_env python -m pip install --no-cache-dir diffusers[torch]==0.15.1 opencv-python==4.7.0.72 \
    pycocotools==2.0.6 matplotlib==3.5.3 ultralytics \
    onnxruntime==1.14.1 onnx==1.13.1 ipykernel==6.16.2 scipy gradio openai

WORKDIR /home/${USER}/Grounded-Segment-Anything
RUN conda run -n cg_env python -m pip install --upgrade setuptools

RUN git clone https://github.com/xinyu1205/recognize-anything.git
RUN conda run -n cg_env python -m pip install -r ./recognize-anything/requirements.txt
RUN conda run -n cg_env python -m pip install -e ./recognize-anything/

RUN conda run -n cg_env python -m pip install supervision==0.19.0
RUN conda run -n cg_env python -m pip install plyfile

ARG UID=1000
ARG GID=1000
RUN useradd -m ${USER} --uid=${UID} \
    && usermod -s /bin/bash ${USER} \
    && usermod -a -G sudo ${USER} \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
WORKDIR /home/${USER}/ConceptGraphs
RUN chown -R $USER /home/${USER}
USER ${USER}

RUN echo 'source activate cg_env' >> /home/${USER}/.bashrc
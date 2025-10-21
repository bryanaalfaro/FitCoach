
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

RUN apt update && \
    apt-get install --yes apt-utils && \
    DEBIAN_FRONTEND=noninteractive apt install --yes --no-install-recommends \
    ffmpeg \
    htop \
    less \
    git \
    make \
    vim \
    wget \
    sudo \
&& rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/opt/conda/

# Install the NCCL development headers and the library itself for some DeepSpeed operations
ADD https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb /tmp/cuda-keyring.deb
RUN sudo dpkg -i /tmp/cuda-keyring.deb && \
    sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends libnccl2 libnccl-dev

# cuda toolkit + link CUDA libraries CUDART and CURAND for the quantization optimizations in DeepSpeed
RUN conda update -n base -c defaults conda && \
    conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit && \
    ln -s /opt/conda/lib/libcudart.so.11.0 /usr/lib/libcudart.so && \
    ln -s /opt/conda/lib/libcurand.so.10 /usr/lib/libcurand.so

# base conda installations
RUN conda install -c conda-forge wheel setuptools==69.5.1 packaging pytest-console-scripts mpi4py
RUN conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
RUN pip install pyarrow==14.0.1
# base pip installations
COPY docker/requirements-pip.txt requirements-pip.txt
RUN pip install --no-cache-dir --upgrade -r requirements-pip.txt
ENV PYTHONBREAKPOINT=ipdb.set_trace

COPY . /fitcoach
WORKDIR /fitcoach

ENV PYTHONPATH="${PYTHONPATH}:/fitcoach/"
ENV PATH="/opt/conda/bin/:${PATH}"


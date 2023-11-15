FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]

USER root

WORKDIR /root

ENV DEBIAN_FRONTEND=noninteractive

# Set env for torch (compute capability)
ENV TORCH_CUDA_ARCH_LIST=9.0

# Set env for huggingface offline mode
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

# Install packages
RUN apt-get update && \
    apt-get install -y pdsh vim openssh-server net-tools libnvidia-compute-515 && \
    mkdir -p /var/run/sshd

# Set for ssh
RUN mkdir .ssh
COPY key.pem .ssh/key.pem
COPY authorized_keys .ssh/authorized_keys

# Install mlnx ofed
COPY MLNX_OFED_LINUX-5.8-3.0.7.0-ubuntu22.04-x86_64 MLNX_OFED_LINUX-5.8-3.0.7.0-ubuntu22.04-x86_64
RUN ./MLNX_OFED_LINUX-5.8-3.0.7.0-ubuntu22.04-x86_64/mlnxofedinstall --user-space-only --without-fw-update -q

# Install nccl
ENV NCCL_DEBUG=INFO
# ENV NCCL_DEBUG_FILE=mnt/output/nccl-debug.%p.log
# ENV NCCL_TOPO_DUMP_FILE=mnt/output/nccl-topo.%p.xml
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update && \
    apt-get install --allow-change-held-packages -y libnccl2=2.18.3-1+cuda12.0 libnccl-dev=2.18.3-1+cuda12.0

# Set hpc-x
ENV HPCX_HOME=/usr/local/hpcx-v2.16-gcc-mlnx_ofed-ubuntu22.04-cuda12-gdrcopy2-nccl2.18-x86_64
COPY hpcx-v2.16-gcc-mlnx_ofed-ubuntu22.04-cuda12-gdrcopy2-nccl2.18-x86_64 /usr/local/hpcx-v2.16-gcc-mlnx_ofed-ubuntu22.04-cuda12-gdrcopy2-nccl2.18-x86_64

# Install python & pip and Install libraries
RUN apt-get install -y curl python3.10-dev python3.10-distutils && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    pip install nvitop==1.3.1 && \
    pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda120

# Copy files that required for training
COPY main.py main.py
COPY pytorch-resnet50.rst pytorch-resnet50.rst

# For util
RUN echo "alias n='nvitop --colorful'" >> ~/.bashrc
RUN apt-get install unzip

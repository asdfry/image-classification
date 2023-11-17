FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]

USER root

WORKDIR /root

ENV DEBIAN_FRONTEND=noninteractive

# Set env for torch (compute capability)
ENV TORCH_CUDA_ARCH_LIST=9.0

# Install packages
RUN apt-get update && \
    apt-get install -y pdsh vim openssh-server net-tools git tmux unzip libnvidia-compute-515 && \
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
ENV NCCL_DEBUG_FILE=/root/mnt/output/nccl-debug.log
ENV NCCL_TOPO_DUMP_FILE=/root/mnt/output/nccl-topo.xml
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update && \
    apt-get install --allow-change-held-packages -y libnccl2=2.18.3-1+cuda12.0 libnccl-dev=2.18.3-1+cuda12.0

# Set hpc-x
ENV HPCX_HOME=/usr/local/hpcx-v2.16-gcc-mlnx_ofed-ubuntu22.04-cuda12-gdrcopy2-nccl2.18-x86_64
COPY hpcx-v2.16-gcc-mlnx_ofed-ubuntu22.04-cuda12-gdrcopy2-nccl2.18-x86_64 /usr/local/hpcx-v2.16-gcc-mlnx_ofed-ubuntu22.04-cuda12-gdrcopy2-nccl2.18-x86_64

# Install python & pip and Install libraries
COPY requirements.txt requirements.txt
RUN apt-get install -y curl python3.10-dev python3.10-distutils && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda120

# Install git lfs
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install git-lfs

# Create symbolic link
RUN ln -s /root/mnt/datasets/imagenet/train train && ln -s /root/mnt/datasets/imagenet/val val

# Copy files that required for training
COPY main.py main.py
COPY utils.py utils.py

# For util
RUN echo "alias n='nvitop --colorful'" >> ~/.bashrc

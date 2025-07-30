FROM nvidia/cuda:12.4.1-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

ARG USER_ID
ARG GROUP_ID

# Use bash shell for conda activation
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y \
    software-properties-common \
    gnupg wget curl git \
    lsb-release ca-certificates \
    sudo time vim
RUN add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update
RUN apt-get install -y gcc-13 g++-13

RUN groupadd -g ${GROUP_ID} myuser \
    && useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash myuser \
    && usermod -aG sudo myuser \
    && echo "myuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# ngAP
RUN apt-get install -y libtbb-dev=2020.1-2
ENV PATH="/opt/conda/bin:${PATH}"
RUN wget https://github.com/Kitware/CMake/releases/download/v3.24.1/cmake-3.24.1-Linux-x86_64.sh \
      -q -O /tmp/cmake-install.sh \
      && chmod u+x /tmp/cmake-install.sh \
      && mkdir /opt/cmake-3.24.1 \
      && /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.24.1 \
      && rm /tmp/cmake-install.sh \
      && ln -s /opt/cmake-3.24.1/bin/* /usr/local/bin

# Hyperscan
RUN apt-get install -y ragel nasm libsqlite3-dev pkg-config libgraphviz-dev
RUN apt-get install -y libboost-all-dev
RUN echo "deb http://dk.archive.ubuntu.com/ubuntu/ xenial main" |  tee -a /etc/apt/sources.list
RUN echo "deb http://dk.archive.ubuntu.com/ubuntu/ xenial universe" |  tee -a /etc/apt/sources.list
RUN apt-get update && apt-get install -y g++-5 gcc-5 
RUN rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100

# BitGen
ENV PATH="/opt/conda/bin:${PATH}"
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
    && rm Miniconda3-latest-Linux-x86_64.sh \
    && chown -R myuser:myuser /opt/conda
RUN conda config --set always_yes yes \
    && conda config --set channel_priority strict \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN conda create -n bitgen-py310 python=3.10 -y \
    && source /opt/conda/etc/profile.d/conda.sh \
    && conda activate bitgen-py310 \
    && conda config --add channels conda-forge \
    && conda install -y cuda-cccl cuda-version=12.4 \
    && pip install typing-extensions --index-url https://pypi.org/simple \
    && pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 \
    && pip install cuda-python==12.8 numpy scipy matplotlib networkx colorama pandas bitarray tqdm torchviz pynvml colorlog pyyaml pygraphviz \
    && pip install git+https://github.com/getianao/figurePlotter.git@2c65e40a8f017fbde058a6fe09b315ee64da4301

RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate bitgen-py310" >> /home/myuser/.bashrc
RUN echo "export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6" >> /home/myuser/.bashrc

CMD ["bash", "--login"]

ENV PYTHONUNBUFFERED=1

ENV BITGEN_ROOT=/BitGen
ENV PATH=${BITGEN_ROOT}/parabix-devel/build:${PATH}
ENV PATH=${BITGEN_ROOT}/hscompile/build:${PATH}
ENV PATH=${BITGEN_ROOT}/ngAP/code/build/bin:${PATH}
ENV PATH=${BITGEN_ROOT}/VASim:${PATH}
ENV CUDA_HOME=/usr/local/cuda

USER myuser
WORKDIR /BitGen
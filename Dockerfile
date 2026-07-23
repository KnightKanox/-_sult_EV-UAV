FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    git \
    libsparsehash-dev \
    build-essential \
    cmake \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

RUN pip3 install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip3 install \
    numpy==1.21.6 \
    matplotlib==3.5.3 \
    scipy==1.7.3 \
    tensorboardx==2.6.2 \
    pyyaml==6.0 \
    opencv-python==4.8.0.76 \
    numba==0.56.4

RUN pip3 install spconv-cu111==2.1.21

WORKDIR /workspace

COPY . .

RUN cd lib/hais_ops && python setup.py build_ext develop

RUN pip3 install -e .

WORKDIR /workspace

CMD ["bash"]
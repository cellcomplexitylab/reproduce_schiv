# Build and run as shown below.
# docker build -t schiv .
# docker run --rm --gpus all -v $(pwd):/tmp -u$(id -u):$(id -g) schiv make

FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

RUN apt-get update -y
RUN apt-get install -y wget

# Install R.
RUN DEBIAN_FRONTEND=noninteractive TZ="America/New_York" apt-get install -y \
	r-base-core=3.6.3-2 r-base-dev=3.6.3-2 \
	libfreetype6-dev libcurl4-openssl-dev libxml2-dev

# Install Python and PIP.
RUN apt-get install -y python3 python3-pip && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN ln /usr/bin/python3 /usr/bin/python

# Install Pandas 
RUN pip3 install pandas

# Install PyTorch and Pyro
RUN pip3 install torch
RUN pip3 install pyro-ppl

# Install R packages
RUN R -e "install.packages('showtext', dependencies=TRUE)"

# Install Salmon
RUN wget https://github.com/COMBINE-lab/salmon/releases/download/v1.5.1/salmon-1.5.1_linux_x86_64.tar.gz
RUN tar xzvf salmon-1.5.1_linux_x86_64.tar.gz && \
	mv salmon-1.5.1_linux_x86_64 /etc/ && \
	ln -s /etc/salmon-1.5.1_linux_x86_64/bin/salmon /usr/bin/ && \
	rm salmon-1.5.1_linux_x86_64.tar.gz

# Use /tmp as shared volume.
WORKDIR /tmp

# Build and run as shown below.
# docker build -t schiv .
# docker run --rm -v $(pwd):/tmp -u$(id -u):$(id -g) schiv make

FROM ubuntu:focal

RUN apt-get update -y

# Install R.
RUN DEBIAN_FRONTEND=noninteractive TZ="America/New_York" apt-get install -y \
	r-base-core=3.6.3-2 r-base-dev=3.6.3-2 \
	libfreetype6-dev libcurl4-openssl-dev libxml2-dev

# Install Python and PIP.
RUN apt-get install -y python3 python3-pip && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN ln /usr/bin/python3 /usr/bin/python

# Install PyTorch and Pyro
RUN pip3 install torch==1.9.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install pyro-ppl==1.7.0

# Insall R packages
RUN R -e "install.packages('showtext', dependencies=TRUE)"

# Use /tmp as shared volume.
WORKDIR /tmp

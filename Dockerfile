FROM python:3.10
# FROM python:3.10-alpine
#3.10-slim
# Based on
# https://github.com/dporkka/docker-101?ysclid=m60umny7nw465329032
# https://hub.docker.com/_/python

# Install system dependencies
RUN apt-get update
RUN apt-get install -y git python3-pip \
        && rm -rf /var/lib/apt/lists/*
        
# for alpine
# RUN apk update 
# RUN apk add git
# RUN apk add sdl2-dev
# RUN apk add py3-pip && pip3 install --upgrade pip
# RUN pip3 install --upgrade wheel 
# RUN pip3 install --upgrade setuptools

# Install any python packages you need
COPY requirements.txt ./
RUN python3 -m pip install --no-cache-dir -r requirements.txt --extra-index-url https://www.wheelodex.org/projects/pygame/ --extra-index-url https://alpine-wheels.github.io/index 

# Upgrade pip
# RUN python3 -m pip install --upgrade pip

# Install PyTorch CPU only
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# Set the working directory
WORKDIR /app

RUN git clone https://github.com/alena284772/hexapod-RL

WORKDIR /app/hexapod-RL

# Set the entrypoint
CMD ["bash"]
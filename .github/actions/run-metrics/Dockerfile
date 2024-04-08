FROM nvidia/cuda:11.0.3-base-ubuntu20.04

# Необходимые пакеты
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    wget \
    unzip 
    
# Install python packages
COPY requirements.txt requirements.txt
# Upgrade pip
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt

VOLUME $GITHUB_WORKSPACE/pull-request-data:$GITHUB_WORKSPACE/pull-request-data

# Install PyTorch
RUN pip3 install tensorrt==8.6.1

# Очистка кэша apt-get
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
COPY main.py /main.py

CMD ["python3", "/main.py"]
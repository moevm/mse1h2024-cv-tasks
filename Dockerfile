FROM python:3.8-slim

# Необходимые пакеты
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-dev \
    libopenblas-dev \
    libjpeg-dev \
    wget \
    unzip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# opencv, PIL, pytorch-gpu, onnx
RUN pip install pip install opencv-python pillow \
    && pip install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html \
    && onnx

# tensor-rt
RUN mkdir /tmp/tensorrt && cd /tmp/tensorrt && \
    wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb && \
    dpkg -i nvidia-machine-learning-repo-*.deb && \
    apt-get update && \
    apt-get install -y libnvinfer7=7.x.y-z+cuda10.x libnvinfer-dev=7.x.y-z+cuda10.x libnvinfer-plugin7=7.x.y-z+cuda10.x

WORKDIR /app

# kaggle 
ARG DATASET
RUN wget -O dataset.zip $DATASET \
    && unzip dataset.zip \
    && rm dataset.zip
    
# Для примера
ADD main.py . 
CMD ["python", "main.py"]

# для сборки:
# docker build --build-arg DATASET=<some_kaggle_dataset_url> -t base-container .


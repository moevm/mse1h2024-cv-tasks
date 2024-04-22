FROM nvidia/cuda:11.0.3-base-ubuntu20.04

# Install python packages
COPY requirements.txt /requirements.txt

COPY entrypoint.sh /entrypoint.sh

# Необходимые пакеты
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \

    && python3 -m pip install --upgrade pip \
    && python3 -m pip install -r /requirements.txt \
    && pip install gdown \
    && pip install --upgrade gdown \
    && apt-get clean && rm -rf /var/lib/apt/lists/* # Очистка кэша apt-get


RUN pip install gdown
RUN pip install --upgrade gdown




COPY links /links
# Очистка кэша apt-get
RUN apt-get clean && rm -rf /var/lib/apt/lists/*


ENTRYPOINT ["sh", "/entrypoint.sh"]

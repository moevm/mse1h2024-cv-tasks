FROM nvidia/cuda:11.0.3-base-ubuntu20.04

# Install python packages
COPY requirements.txt /requirements.txt

# Необходимые пакеты
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    unzip \
    && python3 -m pip install --upgrade pip \
    && python3 -m pip install -r /requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/* # Очистка кэша apt-get

#RUN gdown --fuzzy https://drive.google.com/file/d/15zcMiOwd_aOwdZovyO68gb-dzIjopR6j/view?usp=drive_link -O $GITHUB_WORKSPACE/lab1_dataset.zip
RUN gdown --fuzzy https://drive.google.com/file/d/1pojFjRZuPYyGwFp2wfjX8fgG-yX2aAbK/view?usp=sharing -O $GITHUB_WORKSPACE/lab2_dataset.zip

COPY entrypoint.sh /entrypoint.sh

ENTRYPOINT ["sh", "/entrypoint.sh"]

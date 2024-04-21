FROM nvidia/cuda:11.0.3-base-ubuntu20.04
# Install python packages
COPY requirements.txt /requirements.txt

# Необходимые пакеты
RUN apt-get update \
    && apt-get install -y \
    python3.8 \
    python3-pip \
    && python3 -m pip install --upgrade pip \
    && python3 -m pip install -r /requirements.txt \
    && pip install gdown \
    && pip install --upgrade gdown \
    && apt-get clean && rm -rf /var/lib/apt/lists/* # Очистка кэша apt-get

COPY model_link.txt /model_link.txt
RUN gdown --fuzzy $(cat /model_link.txt) -O /student_file.pth
ENTRYPOINT ["/entrypoint.sh"]



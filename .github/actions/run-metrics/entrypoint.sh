#!/bin/bash

# Функция для обхода папок
traverse() {
    for dir in `find "$1"  -type d -name src`; do
        # Заходим в папку src и скачиваем веса из файла weights_link.txt, если он там есть
          if [ -f "$dir/weights_link.txt" ]; then
            gdown --fuzzy $(cat ${dir}/weights_link.txt) -O ${dir}/weights.pth
          fi
    done
}

# Вызываем функцию для обхода и поиска файла
traverse $GITHUB_WORKSPACE/pull-request-data
python main.py

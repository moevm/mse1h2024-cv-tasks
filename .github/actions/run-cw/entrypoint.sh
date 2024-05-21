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

mkdir $GITHUB_WORKSPACE/$ACTION_WORKSPACE/src/action
mkdir $GITHUB_WORKSPACE/$ACTION_WORKSPACE/src/action/datasets
#unzip /cw1_dataset.zip -d $GITHUB_WORKSPACE/$ACTION_WORKSPACE/src/action/cw1_dataset > trash_tmp
unzip /cw2_dataset.zip -d $GITHUB_WORKSPACE/$ACTION_WORKSPACE/src/action/cw2_dataset > trash_tmp
#unzip /cw3_dataset.zip -d $GITHUB_WORKSPACE/$ACTION_WORKSPACE/src/action/cw3_dataset > trash_tmp

cd $GITHUB_WORKSPACE/$ACTION_WORKSPACE/src

cp -r $GITHUB_WORKSPACE/pull-request-data . 

python3 main.py
# tail -f /dev/null

#!/bin/bash

cd $GITHUB_WORKSPACE/pull-requests-data

gdown --fuzzy $(cat /model_link.txt) -O /student_file.pth

python main.py

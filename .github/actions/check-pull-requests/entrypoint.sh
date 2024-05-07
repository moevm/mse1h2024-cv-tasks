#!/bin/bash

cd $GITHUB_WORKSPACE/$ACTION_WORKSPACE

gh pr list --json number --json title --json files --json labels --state 'open' > opened_pull_requests.json

python main.py

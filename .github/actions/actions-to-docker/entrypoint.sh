#!/bin/bash

cd $GITHUB_WORKSPACE/$ACTION_WORKSPACE

mkdir ~/.ssh

echo $PRIVATE_SSH_KEY | sed 's/\\n/\n/g' > ~/.ssh/id_rsa
chmod 600 ~/.ssh/id_rsa
echo $PUBLIC_SSH_KEY > ~/.ssh/id_rsa.pub
ssh-keyscan github.com > ~/.ssh/known_hosts

git stash

python actions_to_docker.py

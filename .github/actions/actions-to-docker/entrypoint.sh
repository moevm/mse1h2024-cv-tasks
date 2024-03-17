#!/bin/bash

cd $GITHUB_WORKSPACE/$ACTION_WORKSPACE

mkdir ~/.ssh

ssh-keyscan github.com >> ~/.ssh/known_hosts

# python actions_to_docker.py

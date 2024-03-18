#!/bin/bash

directory="./.github/actions"

begin_ssh_private="PRIVATE_SSH_KEY=\""
end="\""
begin_ssh_public="PUBLIC_SSH_KEY=\""
begin_github_token="GITHUB_TOKEN=\""

ssh_private=$(sudo cat ~/.ssh/id_rsa | sed ':a;N;$!ba;s/\n/\\\\n/g')
ssh_public=$(sudo cat ~/.ssh/id_rsa.pub)
github_token=$(gh auth token)

echo $begin_github_token$github_token$end > ./.github/actions/.secrets
echo $begin_ssh_public$ssh_public$end >> ./.github/actions/.secrets
echo $begin_ssh_private$ssh_private$end >> ./.github/actions/.secrets

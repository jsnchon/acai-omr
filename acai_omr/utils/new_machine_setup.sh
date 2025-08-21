#!/bin/bash

# USAGE
# pass in the acai-omr root dir as an argument 

project_dir=$1

echo "Installing poetry"
curl -sSL https://install.python-poetry.org | python3 -

echo "Installing python 3.12"
sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt update && sudo apt install -y python3.12

echo "Installing poetry dependencies"
(cd $project_dir && poetry env use "$(which python3.12)" && poetry install)
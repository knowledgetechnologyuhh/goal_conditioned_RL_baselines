#!/usr/bin/env bash
sudo apt-get install software-properties-common python-software-properties
sudo add-apt-repository ppa:jonathonf/python-3.6
sudo apt-get update
sudo apt-get install python3.6
python3.6 -v

virtualenv -p python3.6 venv
source venv/bin/activate
pip install -r requirements_gpu.txt



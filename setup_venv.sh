#!/bin/bash
rm -rf venv
virtualenv venv -p python3
source venv/bin/activate
pip3 install -r requirements_gpu.txt

#!/bin/bash
apt-get update
apt-get install -y libgl1 libspatialindex-dev
pip install --upgrade pip
pip install -r requirements.txt
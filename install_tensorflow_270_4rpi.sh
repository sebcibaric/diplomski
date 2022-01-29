#! /usr/bin/bash

sudo apt-get update
sudo apt-get upgrade

sudo apt-get install python-pip python3-pip

sudo pip uninstall tensorflow
sudo pip3 uninstall tensorflow

pip3 list | grep numpy

sudo -H pip3 install numpy==1.19.5

sudo apt-get install gfortran
sudo apt-get install libhdf5-dev libc-ares-dev libeigen3-dev
sudo apt-get install libatlas-base-dev libopenblas-dev libblas-dev
sudo apt-get install liblapack-dev

sudo -H pip3 install --upgrade setuptools
sudo -H pip3 install pybind11
sudo -H pip3 install Cython

sudo -H pip3 install h5py==3.1.0

pip3 install gdown
gdown https://drive.google.com/u/0/uc?id=1zf4NasP-rB0FfaQ0DFJxsky04d-Ang9R

sudo -H pip3 install tensorflow-2.7.0-cp37-cp37m-linux_aarch64.whl
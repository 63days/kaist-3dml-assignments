#!/bin/bash

sudo apt-get update
sudo apt-get install -y build-essential \
    software-properties-common

# install cuda first.
wget https://developer.download.nvidia.com/compute/cuda/11.1.0/local_installers/cuda_11.1.0_455.23.05_linux.run
chmod +x cuda_11.1.0_455.23.05_linux.run
sudo ./cuda_11.1.0_455.23.05_linux.run --silent --override-driver-check

echo -e "\n## CUDA and cuDNN paths" >> ~/.bashrc
echo 'export PATH=/usr/local/cuda-11.1/bin:${PATH}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:${LD_LIBRARY_PATH}' >> ~/.bashrc

source ~/.bashrc

# install nvidia-driver. It will reboot the machine.
sudo add-apt-repository -y ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install -y nvidia-driver-460
sudo reboot

# FYI, reinstall driver
# sudo apt-get remove --purge '^nvidia-.*'
# sudo apt-get install -y nvidia-driver-460

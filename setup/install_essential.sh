#!/bin/bash

sudo apt-get update && sudo apt-get install -y \
  apt-utils \
  build-essential \
  curl \
  git \
  htop \
  vim \
  nano \
  sudo \
  cmake \
  software-properties-common \
  libgl1-mesa-glx \
  wget \
  locales \
  tmux \
  apt-utils && \
  sudo apt-get -y autoremove && sudo apt-get -y clean

# Rebuild cache
cd /var/lib/apt && \
  mv lists lists.old && \
  mkdir -p lists/partial && \
  sudo apt-get update && \
  sudo apt-get -y autoremove && sudo apt-get -y clean && \
  cd ~

# Set the locale
sudo locale-gen en_US.UTF-8
echo LANG en_US.UTF-8
echo LANGUAGE en_US:en
echo LC_ALL en_US.UTF-8

# Non-interactive package installation
echo DEBIAN_FRONTEND=noninteractive

sudo apt-get update && sudo apt-get install -y \
        python3.8 \
        python3.8-dev \
        python3-pip \
        virtualenv \
        && \
    sudo apt-get -y autoremove && sudo apt-get -y clean

virtualenv --python=python3.8 ~/venv

echo "source ~/venv/bin/activate" | tee -a ~/.profile
. ~/.profile

pip install -r requirements.txt
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
python --version
python -c "import torch; from pytorch3d.loss.chamfer import chamfer_distance; print(torch.__version__); print(torch.cuda.is_available()); torch.manual_seed(0); print(chamfer_distance(torch.rand(1,1,3).cuda(), torch.rand(1,1,3).cuda())[0]);"

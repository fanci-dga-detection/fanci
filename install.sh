#!/bin/bash
sudo apt-get -y install tmux tshark

curl -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b
echo 'export PATH=$PATH:~/miniconda3/bin/' >> ~/.bashrc
. ~/.bashrc
export PATH=$PATH:~/miniconda3/bin/

conda env create --prefix ~/miniconda3/envs/master_thesis -f environment.yml
mkdir ~/.ipython
cp -r ipython/* ~/.ipython

echo "If you want to use ipython use the already setup profile mt. Start ipython using this cmd: ipython --profile mt"
echo "You have to adjust the workspace root path in the settings.py module."
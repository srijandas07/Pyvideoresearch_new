#/bin/bash
source ~/.bashrc

conda activate torch_4
sudo mountimg /data/stars/share/charades/images-nocomp.squashfs /data/stars/share/charades/test
export PYTHONPATH=../:$PYTHONPATH
python resnet50_3d_charades_test.py

#!/usr/bin/env python
# fine-tune Kinetics-pretrained 3D ResNet50 pretrained on Charades
# original name: i3d12b2
# model_best.txt:
#     CharadesmAPvalvideo 0.31270594963775783
#     loss_train 0.05916500749547305
#     loss_val 0.10518467018531787
#     top1train 42.33025029675106
#     top1val 39.729729729729726
#     top5train 121.67071984435681
#     top5val 133.51351351351352
#     videotop1valvideo 63.66076221148685
#     videotop5valvideo 231.07890499194846
import sys
import pdb
import traceback
sys.path.insert(0, '.')
from main import main
from bdb import BdbQuit
import os
os.nice(19)
name = __file__.split('/')[-1].split('.')[0]

args = [
    '--name', 'test_resnet3d_50',  # name is filename
    '--print-freq', '1',
    '--dataset', 'charades_video',
    '--arch', 'resnet50_3d',
    '--lr', '.375',
    '--criterion', 'default_criterion',
    '--wrapper', 'default_wrapper',
    '--lr-decay-rate', '15,40',
    '--epochs', '2',
    '--batch-size', '8',
    '--video-batch-size', '8',
    '--train-size', '0.0',
    '--weight-decay', '0.0000001',
    '--val-size', '1.0',
    '--cache-dir', '/data/stars/share/charades/cache/',
    '--data', '/data/stars/share/charades/test',
    '--train-file', '/data/stars/share/charades/Charades/Charades_v1_train.csv',
    '--val-file', '/data/stars/share/charades/Charades/Charades_v1_test.csv',
    '--pretrained',
    '--pretrained-weights','/data/stars/share/charades/pretrained/resnet50_rgb_python3.pth.tar',
    '--start-epoch', '1',
    '--workers', '12',
    '--disable-cudnn-benchmark',
    '--disable-cudnn',
    '--replace-last-layer',
    '--tasks', 'video_task',
    '--metric', 'video_task_CharadesmAP',
]
sys.argv.extend(args)
try:
    main()
except BdbQuit:
    sys.exit(1)
except Exception:
    traceback.print_exc()
    print('')
    #pdb.post_mortem()
    #sys.exit(1)

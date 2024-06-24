#!/bin/bash

# 가상 환경 활성화 (가상 환경이 있는 경우)
# source path/to/your/virtualenv/bin/activate

# 학습 스크립트 실행
CUDA_LAUNCH_BLOCKING=1 python3 src/train.py --data_dir preprocessed_coco/ --checkpoint_dir checkpoints/
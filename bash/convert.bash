#!/bin/bash

# 定义输入文件夹路径
ROOT_FOLDER="/root/autodl-tmp/gaussian-splatting/blender/hotdog_FixLight"
DATA_FOLDER="$ROOT_FOLDER/data"
INPUT_FOLDER="$ROOT_FOLDER/data/image"
OUTPUT_FOLDER="$ROOT_FOLDER/out"

TMP_FOLDER="/root/autodl-tmp/code"

python $TMP_FOLDER/suf_convert.py --input_folder $INPUT_FOLDER --output_folder $INPUT_FOLDER
# python convert.py -s $DATA_FOLDER
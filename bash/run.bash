# 定义输入文件夹路径
IMAGE_FOLDER="/root/autodl-tmp/blender_code/shadow_neus/blend_files/point_light/hotdog_specular_point/train_SUN/image"
ROOT_FOLDER="/root/autodl-tmp/gaussian-splatting/blender/hotdog_FixLight3030"
DATA_FOLDER="$ROOT_FOLDER"
INPUT_FOLDER="$ROOT_FOLDER/data/images"
OUTPUT_FOLDER="$ROOT_FOLDER/out"

TMP_FOLDER="/root/autodl-tmp/code"

if [ "$1" = "convert" ]; then
    # 如果输入文件夹不存在，则创建
    if [ ! -d "$INPUT_FOLDER" ]; then
        mkdir -p $INPUT_FOLDER
    fi
    cp -r $IMAGE_FOLDER/* $INPUT_FOLDER
    python $TMP_FOLDER/suf_convert.py --input_folder $INPUT_FOLDER --output_folder $INPUT_FOLDER
elif [ "$1" = "train" ]; then
    python train.py -s $DATA_FOLDER -m $OUTPUT_FOLDER --eval
    python render.py -m $OUTPUT_FOLDER
else 
    echo "Usage: bash run.bash [convert|train]"
fi
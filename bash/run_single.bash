ROOT_FOLDER="/root/autodl-tmp/gaussian-splatting/blender/para_test/hotdog_specular_point"
DATA_FOLDER="$ROOT_FOLDER"
INPUT_FOLDER="$ROOT_FOLDER/data/images"
OUTPUT_FOLDER="$ROOT_FOLDER/out/version0_1223"

python train.py -s $DATA_FOLDER -m $OUTPUT_FOLDER --eval
python render.py -m $OUTPUT_FOLDER
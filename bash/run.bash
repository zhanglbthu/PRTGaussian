# 定义输入文件夹路径
# IMAGE_FOLDER="/root/autodl-tmp/blender_code/shadow_neus/blend_files/point_light/hotdog_specular_point/train_SUN/image"
# ROOT_FOLDER="/root/autodl-tmp/gaussian-splatting/blender/para_test/hotdog_specular_point"
# DATA_FOLDER="$ROOT_FOLDER"
# INPUT_FOLDER="$ROOT_FOLDER/data/images"
# OUTPUT_FOLDER="$ROOT_FOLDER/test/test_1218aft00"

ROOT_FOLDER="/root/autodl-tmp/gaussian-splatting/blender/para_test"

objs=(chair_specular_point drums_specular_point ficus_specular_point hotdog_specular_point lego_specular_point materials_specular_point mic_specular_point ship_specular_point)
gpus=(0 1 2 3 4 5 6 7)

cur_idx=0
tmux_name=run_gs
tmux new-session -d -s ${tmux_name}

VERSION=diffuse_121800

for obj in ${objs[@]}; do
    ((cur_idx=cur_idx+1))
    echo "${cur_idx}, ${obj}"
    DATA_FOLDER="$ROOT_FOLDER/$obj"
    OUTPUT_FOLDER="$ROOT_FOLDER/$obj/out/$VERSION"
    tmux new-window -t ${tmux_name}:${cur_idx} -n ${obj}
    tmux send-keys -t ${tmux_name}:${cur_idx} "conda activate gaussian_splatting" ENTER
    tmux send-keys -t ${tmux_name}:${cur_idx} "export CUDA_VISIBLE_DEVICES=${gpus[cur_idx-1]}" ENTER
    tmux send-keys -t ${tmux_name}:${cur_idx} "python train.py -s $DATA_FOLDER -m $OUTPUT_FOLDER --eval" ENTER
    tmux send-keys -t ${tmux_name}:${cur_idx} "python render.py -m $OUTPUT_FOLDER" ENTER
done

# python train.py -s $DATA_FOLDER -m $OUTPUT_FOLDER --eval
# python render.py -m $OUTPUT_FOLDER
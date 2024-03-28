#!/bin/bash
blender_path="/root/blender-3.6.5-linux-x64/blender"
objs=(hotdog_specular_point)
gpus=(0 1 2 3 4 5 6 7)
source_path="/root/autodl-tmp/blender_code/shadow_neus/blend_files"

root_path="/root/autodl-tmp/gaussian-splatting/eval/point_light"

cur_idx=0
tmux_name=directional_light_render
tmux new-session -d -s ${tmux_name}
    
for obj in "${objs[@]}"; do
    ((cur_idx=cur_idx+1))
    echo "${cur_idx}, ${obj}"
    data_path="${root_path}/${obj}"
    image_path="${data_path}/images"
    video_path="${data_path}/video"
    tmux new-window -t ${tmux_name}:${cur_idx} -n ${obj}
    tmux send-keys -t ${tmux_name}:${cur_idx} "conda activate blender" ENTER
    tmux send-keys -t ${tmux_name}:${cur_idx} "cd ${source_path}" ENTER
    tmux send-keys -t ${tmux_name}:${cur_idx} "cd point_light/${obj}" ENTER
    tmux send-keys -t ${tmux_name}:${cur_idx} "export CUDA_VISIBLE_DEVICES=${gpus[cur_idx-1]}" ENTER
    tmux send-keys -t ${tmux_name}:${cur_idx} "${blender_path} --background --factory-startup main.blend --python ../../360_view_point_train.py -- ${data_path}" ENTER
    tmux send-keys -t ${tmux_name}:${cur_idx} "python ../../image2video.py ${image_path} ${video_path}" ENTER
done

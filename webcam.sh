#!/bin/bash

first_dir="address"
second_dir="result_"
dataset="Privatev1"
_view="all"
ckpt="laneatt_r18_culane"
data_dir="datasets/Privatev1/Input/"
conf_threshold="0.4"
nms_thres="45."
max_lane="2"
webcam="webcam"

for i in $conf_threshold
do for j in $nms_thres
  do for k in $max_lane
    do CUDA_VISIBLE_DEVICES=0 python main.py test --exp_name ${ckpt} --epoch 15 --web ${webcam} --view all --conf_threshold $i --nms_thres $j --max_lane $k --data_dir ${data_dir} --test_dataset ${dataset} --test_first_dir ${first_dir} --test_second_dir ${second_dir}
    done;
  done;
done;

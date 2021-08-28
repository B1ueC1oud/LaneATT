#!/bin/bash

'''
### Hyperparameter
When you test using the **.sh** file, you need to modify the hyperparameter as shown below to run it.
- **first_dir**: First location where output image and video will be saved after testing with KODAS dataset
- **second_dir**: Second location where output image and video will be saved after testing with KODAS dataset next to **first_dir**
- **dataset**: the dataset you will use for testing **choice=[kodasv1,kodasv3]**
- **_view(default: all)**: A hyperparameter used to identify a mistake and save it (KODAS does not have a label, so there is no need to change it)
- **ckpt**: Saved pretrained config file and model weights(Use for directory **experiment/**)
- **data_dir**: Location of KODAS v1 and v3 datasets
- **conf_threshold(default: 0.4)**: laneATT hyperparameter
- **nms_thres(default: 45.)**: laneATT hyperparameter
- **max_lane(default: 2)**: Maximum number of lanes that can be detected
'''

first_dir="/data2/lane_data/ckpt/"
second_dir="result_"
dataset="Private"
_view="all"
ckpt="laneatt_r18_culane"
data_dir="/data2/datasets/Private_Data/Input/"
conf_threshold="0.4"
nms_thres="45."
max_lane="2"

for i in $conf_threshold
do for j in $nms_thres
  do for k in $max_lane
    do CUDA_VISIBLE_DEVICES=0 python main.py test --exp_name ${ckpt} --view all --conf_threshold $i --nms_thres $j --max_lane $k --data_dir ${data_dir} --test_dataset ${dataset} --test_first_dir ${first_dir} --test_second_dir ${second_dir}
    done;
  done;
done;
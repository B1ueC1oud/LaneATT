## KODAS V1, V3 Lane Detection


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

### Example
1. When testing unlabeled datasets other than KODAS v1 and v3 datasets, there are two methods.
    - After setting the hyperparameter **dataset** to **kodasv1**, put the absolute address to the folder containing **.jpg** in hyperparameter **data_dir**(ex: "/data/lanedata/kodas").
    - Change **"Kodas_LaneATT/lib/datasets/nolabel_dataset.py"** file. (you need to change function **load_annotations**)

2. If you set the hyperparameter as below, the address where the ouput image or video will be saved is **"/data/lane_detection/result_kodasv1/laneatt_r18_culane/conf_threshold_0.4_nms_thres_45.0_max_lane_2/"**.
    - **first_dir**: "/data/lane_detection"
    - **second_dir**: "result_"
    - **dataset**: "kodasv1"
    - **_view(default: all)**: "all"
    - **ckpt**: "laneatt_r18_culane"
    - **conf_threshold(default: 0.4)**: 0.4
    - **nms_thres(default: 45.)**: 45.
    - **max_lane(default: 2)**: 2
3. **It is not a clean code that leaves only what is needed for the test of the kodas dataset on the reference baseline.**
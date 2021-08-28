## No label Data & ADD BDD 100k Datasets for Lane Detection


### Notice
The code corresponding to No label DATA except for the BDD 100k part was written according to specific private data. <br>
In addition, since private data will not be disclosed, related contents and names are hidden. <br>
So there are awkward parts in README.md and README2.md contents.


## Pretrain model 
We pretrain Culane Resnet18 pretrained model, Tusimple Resnet18 pretrained model.
Plus, ADD BDD 100k dataset pretrain models (Comming soon)
See **experiments/** directory


## Main requirements
- Python >= 3.5
- PyTorch == 1.6, tested on CUDA 10.2. The models were trained and evaluated on PyTorch 1.6. When testing with other versions, the results (metrics) are slightly different.
- CUDA, to compile the NMS code (**You MUST do this part see reference git**)
- Other dependencies described in `requirements.txt`

## Install
See, **Reference git page** -> 2. Install part

## Getting start
See, **Reference git page** -> 3. Getting Start

## Dataset
**Culane, Tusimple dataset: see Referce git page(DATASETS.md)**
when you using No label Dataset you need to set 


**BDD 100k**
Coming soon

**Private v1**
```
Privatev1
├── Input
├── Output
└── GT
```

**Private v3** 
```
Private_v3
├── 2019Y07M05D16H49m20s
│   ├── RadarF
│   ├── NsuMDB
│   ├── Object
│   ├── VLS128
│   └── Camera_FrontMid
│        └── FrontMid
├── 2019Y07M05D15H43m44s
│   ├──...
├── 2019Y07M05D16H18m12s
│   ├──...
├── 2019Y07M05D16H47m20s
│   ├──...
├── 2019Y07M05D16H01m58s
│   ├──...
├── 2019Y07M05D15H48m42s
│   ├──...
└── 2019Y07M05D16H41m35s
    ├──...
```

## Usage
```bash

# New Conda Environment
conda create -n laneatt python=3.8 -y
conda activate laneatt
conda install pytorch==1.6 torchvision -c pytorch

#Get pip Environment
pip install -r requirements.txt
#Get Conda Environment(options)
conda env create -f environment.yaml

#NMS setup
cd lib/nms; python setup.py install; cd -

# test Private V1
sh Private.sh
# test Private V3
sh Private2.sh
```

## Public Dataset Results

#### CULane
|   Backbone    |     F1, official implement. (%)    | F1, Ref implement. (%) | F1, Own Test Result. (%) |
|     :---      |         ---:                       |   ---:                 |   ---:                   | 
| ResNet-18     | 75.13                              |  75.08                 |  75.05                   | 
| ResNet-34     | 76.68                              |  76.66                 |  76.26                   |



#### TuSimple
|   Backbone                |      Accuracy (%)     |      FDR (%)     |      FNR (%)     |
|    :---                   |         ---:          |       ---:       |       ---:       |
| ResNet-18                 |    95.57              |    3.56          |    3.01          |
| ResNet-34                 |    95.63              |    3.53          |    2.92          |
| ResNet-18(Own Test Result)|    95.71              |    3.63          |    3.03          |
| ResNet-34(Own Test Result)|    95.59              |    3.42          |    3.11          |

#### BDD 100k


## Private Dataset Result 

### Private Version1 Output Example

[![Private v1 video](data/figures/Private_v1_somenail.PNG "Private v1 video")](https://drive.google.com/file/d/1pFEE4BS-hTz8jQ8ngwvC9eF1RnsaBhNM/view?usp=sharing)

#### Private Version3 Output Example

[![Private v3 video](data/figures/Private_v3_somenail.PNG "Private v3 video")](https://drive.google.com/file/d/19g3bxXVbK9bQHNNodMMGjEmh8tm-R97t/view?usp=sharing)

## Hyper Parameters
See Reference git page or  [README2.md](README2.md).




## Reference
https://github.com/lucastabelini/LaneATT

Thanks for the good baseline code


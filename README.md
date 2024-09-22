<div align="center">
  <img width="125px" src="imgs/logo.png"/>
  
  # Incremental Learning for Robot Shared Autonomy
</div>


<p align="left">
    <a href=''>
      <img src='https://img.shields.io/badge/Paper-arXiv-green?style=plastic&logo=arXiv&logoColor=green' alt='Paper arXiv'>
    </a>
    <a href='https://ilsa-robo.github.io/'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
</p>
This is the official repo for the paper:  


**[Incremental Learning for Robot Shared Autonomy](https://https://ilsa-robo.github.io/)**  
[Yiran Tao](https://yirantao1000.github.io/), [Guuixiu Qiao](https://https://ilsa-robo.github.io/), [Ding Dan](https://www.shrs.pitt.edu/people/dan-ding), [Zackory Erickson](https://zackory.com/)  
in Submission

ILSA is an Incrementally Learned Shared Autonomy framework that continually improves its assistive control policy through repeated user interactions. ILSA leverages synthetic kinematic trajectories for initial pretraining, reducing the need for expert demonstrations, and then incrementally finetunes its policy after each manipulation interaction, with mechanisms to balance new knowledge acquisition with existing knowledge retention during incremental learning.

## Table of Contents
- [Setup](#setup)
  <!-- - [ILSA](#ilsa)
  - [OMPL](#Open-Motion-Planning-Library)
  - [Dataset](#dataset) -->
- [Generate Synthetic Kinematic Trajectories](#generate-synthetic-kinematic-trajectories)
  - [Generate raw trajectories](#generate-raw-trajectories)
  - [Normalization](#normalization)
  - [Convert to hdf5 file](#convert-to-hdf5-file)
- [Pretrain the Action Generation Model](#pretrain-the-action-generation-model)  
- [Run ILSA](#run-ilsa)  
  - [User uses ILSA](#user-uses-ilsa)


## Setup
<!-- ### ILSA -->
Clone this git repo.
```
git clone https://github.com/yirantao1000/ilsa.git
```
We recommend working with a conda environment.
```
conda env create -f environment.yaml
conda activate ilsa
```
If installing from this yaml file doesn't work, manual installation of missing packages should also work.

Part of the code is based on [Robomimic](https://robomimic.github.io/). Install robomimic following [instructions](https://robomimic.github.io/docs/introduction/installation.html).

Object localization of the code is based on [SAM with text prompt](https://github.com/luca-medeiros/lang-segment-anything) implemented by [Luca Medeiros](https://github.com/luca-medeiros). Set up following [instructions](https://github.com/luca-medeiros/lang-segment-anything).

## Generate Synthetic Kinematic Trajectories

### Generate raw trajectories
Fill in the codes the robot initial position, x and y ranges of the robot workspace, and position ranges of the experiment objects. Then run:
```
python generate_traj_cereal.py #cereal puring task

python generate_traj_pill.py #pill bottle storage task
``` 

### Normalization
```
python normalize.py --folder data/pill/raw --target_folder data/pill/normalized
``` 
Paste the printed ```min_proprios```, ```max_proprios```, ```min_actions```, and ```max_actions``` values in the ```find_min_max```function in ```robomimic/ilsa.py```,  ```run_ILSA.py```, and ```generate_corrected_traj.py```.

### Convert to hdf5 file
```
python to_hdf5.py --data_name #cereal or pill 
``` 


## Pretrain the Action Generation Model
This part is partially based on [Robomimic](https://robomimic.github.io/).
Move the scripts below to the corresponding directory within your robomimic installation path:
```
mv robomimic/train_ilsa.py /path/to/robomimic/directory/robomimic/scripts
mv robomimic/ilsa.py /path/to/robomimic/directory/robomimic/algo
mv robomimic/ilsa_config.py /path/to/robomimic/directory/robomimic/config
mv robomimic/obs_nets.py /path/to/robomimic/directory/robomimic/models
mv robomimic/policy_nets.py /path/to/robomimic/directory/robomimic/models
```
Enter your robomimic installation path:
```
cd /path/to/robomimic/directory/
```
Run
```
python robomimic/scripts/train_ilsa.py --name cereal --config /path/to/ilsa/directory/configs/cereal.json --dataset /path/to/ilsa/directory/data/cereal/kinematics.hdf5 --output_dir /path/to/ilsa/directory/exp_results

python robomimic/scripts/train_ilsa.py --name pill --config /path/to/ilsa/directory/configs/pill.json --dataset /path/to/ilsa/directory/data/pill/kinematics.hdf5 --output_dir /path/to/ilsa/directory/exp_results
```
Paste the ```ckpt_path``` into ```run_ILSA.py```, and ```previous_ckpt_path``` in ```incre_pill.json``` or ```incre_milk.json```.

## Run ILSA
Repeat the steps below:
### User uses ILSA
```
python run_ILSA.py
```
### Generate Corrected Trajectory
```
python generate_corrected_traj.py
```
### Finetune ILSA
```
cd /path/to/robomimic/directory/
python robomimic/scripts/train_ILSA.py --name cereal_incre --config  /path/to/ilsa/directory/configs/incre_cereal.json --dataset /path/to/ilsa/directory/data/cereal/incremental/0/modified_0-0+50init.hdf5 --output_dir /path/to/ilsa/directory/exp_results
```
After finetuning, modify ```previous_ckpt_path``` in ```incre_pill.json``` or ```incre_milk.json``` to prepare for the next finetuning.

## Acknowledgements
- Part of the codes is based on [Robomimic](https://robomimic.github.io/)
- Object localization of the code is based on [SAM with text prompt](https://github.com/luca-medeiros/lang-segment-anything) implemented by [Luca Medeiros](https://github.com/luca-medeiros). 

<!-- ## Citation
If you find this codebase/paper useful for your research, please consider citing:
```
@article{wang2023robogen,
  title={Robogen: Towards unleashing infinite data for automated robot learning via generative simulation},
  author={Wang, Yufei and Xian, Zhou and Chen, Feng and Wang, Tsun-Hsuan and Wang, Yian and Fragkiadaki, Katerina and Erickson, Zackory and Held, David and Gan, Chuang},
  journal={arXiv preprint arXiv:2311.01455},
  year={2023}
}
``` -->


